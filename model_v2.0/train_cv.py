"""
Phase 2: 5-fold StratifiedGroupKFold CV training + evaluation.

- Same folds as Phase 1 (groups=request_user_agent, random_state=42)
- Downsamples Normal + SQL Injection to max 100K each in train
- Trains DistilBERT for exactly 4 epochs per fold, no early stopping
- Evaluates on full (non-downsampled) test fold
- Reports per-class F1 mean +/- std across evaluable folds
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
)
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

# ── Inline config (same as v2.0) ──────────────────────────────
BASE_MODEL = "distilbert-base-uncased"
MAX_LENGTH = 256
BATCH_SIZE = 64
LEARNING_RATE = 2e-5
DROPOUT = 0.3
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
SIGMOID_THRESHOLD = 0.5
MAX_BODY_CHARS = 500
N_FOLDS = 5
N_EPOCHS = 4
RANDOM_SEED = 42
MAX_NORMAL_TRAIN = 100_000
MAX_SQLI_TRAIN = 100_000

LABEL_COLS = [
    '000 - Normal',
    '272 - Protocol Manipulation',
    '242 - Code Injection',
    '88 - OS Command Injection',
    '126 - Path Traversal',
    '66 - SQL Injection',
    '16 - Dictionary-based Password Attack',
    '310 - Scanning for Vulnerable Software',
    '153 - Input Data Manipulation',
    '274 - HTTP Verb Tampering',
    '194 - Fake the Source of Data',
    '34 - HTTP Response Splitting',
    '33 - HTTP Request Smuggling',
]

DOWNSAMPLE_CLASSES = {
    '000 - Normal': MAX_NORMAL_TRAIN,
    '66 - SQL Injection': MAX_SQLI_TRAIN,
}


# ── Model ─────────────────────────────────────────────────────
class MultiLabelWAFModel(nn.Module):
    def __init__(self, model_name=BASE_MODEL, num_labels=13, dropout=DROPOUT):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels),
        )
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        return {"logits": logits}

    def get_weighted_loss(self, logits, labels, pos_weight):
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        return loss_fn(logits, labels)


# ── Dataset ───────────────────────────────────────────────────
class WAFDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        print(f"    Pre-tokenizing {len(texts)} samples (max_length={max_length})...")
        batch_size = 4096
        all_ids, all_masks = [], []
        for i in tqdm(range(0, len(texts), batch_size),
                      desc="    Tokenizing", bar_format='{l_bar}{bar:30}{r_bar}'):
            chunk = texts[i:i + batch_size]
            enc = tokenizer(chunk, truncation=True, max_length=max_length,
                            padding="max_length", return_tensors="pt")
            all_ids.append(enc["input_ids"])
            all_masks.append(enc["attention_mask"])
        self.input_ids = torch.cat(all_ids)
        self.attention_masks = torch.cat(all_masks)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        print(f"    Done. Shape: {list(self.input_ids.shape)}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {"input_ids": self.input_ids[idx],
                "attention_mask": self.attention_masks[idx],
                "labels": self.labels[idx]}


# ── Serialization (same as v2.0) ──────────────────────────────
def serialize_row(row):
    """Structured serialization: [METHOD] [PATH] [UA] [CT] [BODY]"""
    method = str(row.get('request_http_method', '')).strip()
    path = str(row.get('request_http_request', '')).strip()
    ua = str(row.get('request_user_agent', '')).strip()
    ct = str(row.get('request_content_type', '')).strip()
    body = str(row.get('request_body', '')).strip()

    if body.lower() in ('nan', '', 'none'):
        body = ''
    if ct.lower() in ('nan', '', 'none'):
        ct = ''
    if ua.lower() in ('nan', '', 'none'):
        ua = ''
    if method.lower() in ('nan', '', 'none'):
        method = ''

    # Truncate body
    if len(body) > MAX_BODY_CHARS:
        body = body[:MAX_BODY_CHARS]

    parts = []
    if method:
        parts.append(f"[METHOD] {method}")
    if path:
        parts.append(f"[PATH] {path}")
    if ua:
        parts.append(f"[UA] {ua}")
    if ct:
        parts.append(f"[CT] {ct}")
    if body:
        parts.append(f"[BODY] {body}")

    return ' '.join(parts) if parts else '[EMPTY]'


# ── Downsampling ──────────────────────────────────────────────
def downsample_train(train_idx, labels_np, rng):
    """Downsample Normal and SQL Injection in train.
    Keeps ALL rows for minority classes."""
    dominant = labels_np[train_idx].argmax(axis=1)
    keep_mask = np.ones(len(train_idx), dtype=bool)

    for class_name, max_rows in DOWNSAMPLE_CLASSES.items():
        class_idx = LABEL_COLS.index(class_name)
        is_dominant = dominant == class_idx
        dominant_count = is_dominant.sum()
        if dominant_count > max_rows:
            dominant_positions = np.where(is_dominant)[0]
            drop_count = dominant_count - max_rows
            drop_positions = rng.choice(dominant_positions, size=drop_count, replace=False)
            keep_mask[drop_positions] = False
            print(f"    Downsampled {class_name}: {dominant_count} -> {dominant_count - drop_count}")

    return train_idx[keep_mask]


# ── Training one fold ─────────────────────────────────────────
def train_fold(fold_id, train_texts, train_labels, test_texts, test_labels,
               label_cols, device, output_dir):
    """Train one fold and return evaluation results."""
    fold_start = time.time()
    num_labels = len(label_cols)

    # Compute pos_weights from downsampled train
    train_pos = train_labels.sum(axis=0)
    train_neg = len(train_labels) - train_pos
    pos_weight = np.clip(train_neg / np.maximum(train_pos, 1), 1.0, 50.0)
    class_weights = torch.tensor(pos_weight, dtype=torch.float32).to(device)

    print(f"\n  Train support per class:")
    for j, col in enumerate(label_cols):
        print(f"    {col:.<50s} {int(train_pos[j]):>6d}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # Datasets
    train_ds = WAFDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    test_ds = WAFDataset(test_texts, test_labels, tokenizer, MAX_LENGTH)

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=2, pin_memory=pin)

    # Model (fresh for each fold)
    model = MultiLabelWAFModel(num_labels=num_labels)
    model.to(device)
    raw_model = model

    if torch.cuda.device_count() > 1:
        print(f"  Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * N_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # ── Train N_EPOCHS (no early stopping, no val) ──
    for epoch in range(N_EPOCHS):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0

        progress = tqdm(train_loader,
                        desc=f"  Fold {fold_id} Epoch {epoch+1}/{N_EPOCHS}",
                        bar_format='{l_bar}{bar:30}{r_bar}', unit='batch')

        for step, batch in enumerate(progress):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(input_ids, attention_mask)
                loss = raw_model.get_weighted_loss(outputs["logits"], labels, class_weights)
                if loss.dim() > 0:
                    loss = loss.mean()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            train_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}",
                                 avg=f"{train_loss/(step+1):.4f}")

        epoch_time = time.time() - epoch_start
        avg_loss = train_loss / len(train_loader)
        print(f"    Epoch {epoch+1} done ({epoch_time:.0f}s), avg loss: {avg_loss:.4f}")

    # ── Evaluate on test fold ──
    print(f"\n  Evaluating fold {fold_id} on test set ({len(test_texts)} samples)...")
    model.eval()
    all_preds, all_labels_list, all_probs = [], [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="  Evaluating",
                          bar_format='{l_bar}{bar:30}{r_bar}'):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids, attention_mask)
            probs = torch.sigmoid(outputs["logits"]).cpu().numpy()
            preds = (probs >= SIGMOID_THRESHOLD).astype(int)
            all_preds.append(preds)
            all_labels_list.append(batch["labels"].numpy().astype(int))
            all_probs.append(probs)

    all_preds = np.concatenate(all_preds)
    all_labels_arr = np.concatenate(all_labels_list)

    # Per-class metrics
    per_class = {}
    evaluable_classes = []
    not_evaluable = []

    for j, col in enumerate(label_cols):
        train_sup = int(train_labels[:, j].sum())
        test_sup = int(all_labels_arr[:, j].sum())

        if train_sup == 0 or test_sup == 0:
            per_class[col] = {
                "precision": None, "recall": None, "f1": None,
                "train_support": train_sup, "test_support": test_sup,
                "evaluable": False,
            }
            not_evaluable.append(col)
        else:
            p = precision_score(all_labels_arr[:, j], all_preds[:, j], zero_division=0)
            r = recall_score(all_labels_arr[:, j], all_preds[:, j], zero_division=0)
            f = f1_score(all_labels_arr[:, j], all_preds[:, j], zero_division=0)
            per_class[col] = {
                "precision": float(p), "recall": float(r), "f1": float(f),
                "train_support": train_sup, "test_support": test_sup,
                "evaluable": True,
            }
            evaluable_classes.append(col)

    # Macro-F1 (evaluable only)
    evaluable_f1s = [per_class[c]["f1"] for c in evaluable_classes]
    macro_f1_evaluable = float(np.mean(evaluable_f1s)) if evaluable_f1s else 0.0
    micro_f1 = f1_score(all_labels_arr, all_preds, average='micro', zero_division=0)

    fold_time = time.time() - fold_start

    result = {
        "fold_id": fold_id,
        "train_size": len(train_texts),
        "test_size": len(test_texts),
        "n_epochs": N_EPOCHS,
        "per_class": per_class,
        "evaluable_classes": evaluable_classes,
        "not_evaluable": not_evaluable,
        "macro_f1_evaluable": macro_f1_evaluable,
        "micro_f1": float(micro_f1),
        "fold_time_seconds": fold_time,
    }

    # Print fold summary
    print(f"\n  {'='*60}")
    print(f"  FOLD {fold_id} RESULTS ({fold_time:.0f}s)")
    print(f"  {'='*60}")
    print(f"  Train: {len(train_texts)}, Test: {len(test_texts)}")
    print(f"  Evaluable: {len(evaluable_classes)}/{len(label_cols)} classes")
    print(f"  Micro-F1: {micro_f1:.4f}")
    print(f"  Macro-F1 (evaluable only): {macro_f1_evaluable:.4f}")
    print(f"\n  Per-class:")
    print(f"  {'Class':<50s} {'P':>6s} {'R':>6s} {'F1':>6s} {'TrSup':>7s} {'TeSup':>7s}")
    print(f"  {'-'*50} {'-'*6} {'-'*6} {'-'*6} {'-'*7} {'-'*7}")
    for col in label_cols:
        pc = per_class[col]
        if pc["evaluable"]:
            print(f"  {col:<50s} {pc['precision']:>6.3f} {pc['recall']:>6.3f} {pc['f1']:>6.3f} "
                  f"{pc['train_support']:>7d} {pc['test_support']:>7d}")
        else:
            print(f"  {col:<50s}   N/A    N/A    N/A "
                  f"{pc['train_support']:>7d} {pc['test_support']:>7d}  <-- NOT EVALUABLE")

    if not_evaluable:
        print(f"\n  Not evaluable in this fold:")
        for c in not_evaluable:
            pc = per_class[c]
            reason = "train=0" if pc["train_support"] == 0 else "test=0"
            print(f"    - {c} ({reason})")

    # Save fold results
    fold_dir = os.path.join(output_dir, "cv")
    os.makedirs(fold_dir, exist_ok=True)
    fold_path = os.path.join(fold_dir, f"fold_{fold_id}.json")
    with open(fold_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved to {fold_path}")

    # Save model checkpoint
    ckpt_path = os.path.join(fold_dir, f"fold_{fold_id}_model.pt")
    torch.save(raw_model.state_dict(), ckpt_path)
    print(f"  Checkpoint: {ckpt_path}")

    # Clean up GPU memory
    del model, raw_model, train_ds, test_ds, train_loader, test_loader
    del optimizer, scheduler, scaler
    torch.cuda.empty_cache()

    return result


# ── Main ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Phase 2: 5-fold CV training")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", default="outputs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}")
    print(f"  Phase 2: 5-Fold StratifiedGroupKFold CV")
    print(f"  Device: {device}, GPUs: {torch.cuda.device_count()}")
    print(f"  Epochs per fold: {N_EPOCHS}, Batch size: {BATCH_SIZE}")
    print(f"  Downsample: Normal<={MAX_NORMAL_TRAIN:,}, SQLi<={MAX_SQLI_TRAIN:,}")
    print(f"{'='*70}")

    # ── Load and preprocess ───────────────────────────────────
    print("\nLoading dataset...")
    df = pd.read_csv(args.data_path, low_memory=False)
    print(f"  Loaded {len(df)} rows")

    # Merge 248 -> 88
    if '248 - Command Injection' in df.columns and '88 - OS Command Injection' in df.columns:
        df['88 - OS Command Injection'] = (
            df['88 - OS Command Injection'].fillna(0).astype(int) |
            df['248 - Command Injection'].fillna(0).astype(int)
        ).astype(int)
        df = df.drop(columns=['248 - Command Injection'])
        print("  Merged CAPEC-248 into CAPEC-88")

    df = df.dropna(subset=['request_http_request']).reset_index(drop=True)
    print(f"  After dropping NaN text: {len(df)} rows")

    # Serialize all rows
    print("  Serializing rows...")
    texts = []
    for _, row in tqdm(df.iterrows(), total=len(df),
                       desc="  Serializing", bar_format='{l_bar}{bar:30}{r_bar}'):
        texts.append(serialize_row(row))
    print(f"  Serialized {len(texts)} rows")

    labels_np = df[LABEL_COLS].fillna(0).astype(int).values

    # Groups + stratification proxy
    groups = df['request_user_agent'].fillna('__MISSING__').values
    y_strat = df[LABEL_COLS].fillna(0).astype(int).idxmax(axis=1).values

    # ── 5-fold CV ─────────────────────────────────────────────
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    rng = np.random.RandomState(RANDOM_SEED)

    all_results = []

    for fold_id, (train_idx, test_idx) in enumerate(sgkf.split(df, y_strat, groups)):
        print(f"\n{'='*70}")
        print(f"  FOLD {fold_id}/{N_FOLDS-1}")
        print(f"{'='*70}")

        # Verify 0% UA overlap
        train_uas = set(groups[train_idx])
        test_uas = set(groups[test_idx])
        overlap = train_uas & test_uas
        assert len(overlap) == 0, f"UA overlap in fold {fold_id}: {len(overlap)}"
        print(f"  UA overlap: 0 (verified)")
        print(f"  Raw train: {len(train_idx)}, Test: {len(test_idx)}")

        # Downsample train
        ds_train_idx = downsample_train(train_idx, labels_np, rng)
        print(f"  Downsampled train: {len(ds_train_idx)}")

        # Build text/label arrays
        train_texts = [texts[i] for i in ds_train_idx]
        train_labels = labels_np[ds_train_idx]
        test_texts_fold = [texts[i] for i in test_idx]
        test_labels = labels_np[test_idx]

        result = train_fold(
            fold_id, train_texts, train_labels,
            test_texts_fold, test_labels,
            LABEL_COLS, device, args.output_dir,
        )
        all_results.append(result)

        # After fold 0, print estimated runtime
        if fold_id == 0:
            est_total = result["fold_time_seconds"] * N_FOLDS
            print(f"\n  *** Estimated total runtime: {est_total/60:.0f} min "
                  f"({est_total/3600:.1f} hours) ***")

    # ── Aggregate results ─────────────────────────────────────
    print(f"\n\n{'='*70}")
    print(f"  AGGREGATE CV RESULTS ({N_FOLDS} folds)")
    print(f"{'='*70}")

    # Per-class F1 mean +/- std across evaluable folds
    print(f"\n  Per-class F1 (mean +/- std across evaluable folds):")
    print(f"  {'Class':<50s} {'Mean F1':>8s} {'Std':>8s} {'Folds':>6s}")
    print(f"  {'-'*50} {'-'*8} {'-'*8} {'-'*6}")

    never_evaluable = []
    class_agg = {}

    for col in LABEL_COLS:
        fold_f1s = []
        for r in all_results:
            pc = r["per_class"][col]
            if pc["evaluable"]:
                fold_f1s.append(pc["f1"])

        if len(fold_f1s) == 0:
            print(f"  {col:<50s}      N/A      N/A    0/{N_FOLDS}  <-- NEVER EVALUABLE")
            never_evaluable.append(col)
        else:
            mean_f1 = np.mean(fold_f1s)
            std_f1 = np.std(fold_f1s)
            print(f"  {col:<50s}   {mean_f1:.4f}   {std_f1:.4f}  {len(fold_f1s):>2d}/{N_FOLDS}")
            class_agg[col] = {"mean_f1": float(mean_f1), "std_f1": float(std_f1),
                              "n_folds": len(fold_f1s), "fold_f1s": fold_f1s}

    # Macro-F1 per fold (evaluable classes only)
    print(f"\n  Macro-F1 per fold (evaluable classes only):")
    fold_macros = []
    for r in all_results:
        print(f"    Fold {r['fold_id']}: {r['macro_f1_evaluable']:.4f} "
              f"({len(r['evaluable_classes'])} evaluable classes)")
        fold_macros.append(r['macro_f1_evaluable'])

    mean_macro = np.mean(fold_macros)
    std_macro = np.std(fold_macros)
    print(f"\n  Mean Macro-F1: {mean_macro:.4f} +/- {std_macro:.4f}")

    # Never evaluable
    if never_evaluable:
        print(f"\n  Classes NEVER evaluable in any fold:")
        for c in never_evaluable:
            print(f"    - {c}")
    else:
        print(f"\n  All classes evaluable in at least one fold.")

    # Save aggregate
    agg = {
        "n_folds": N_FOLDS,
        "n_epochs": N_EPOCHS,
        "mean_macro_f1": float(mean_macro),
        "std_macro_f1": float(std_macro),
        "per_class": {col: class_agg.get(col, {"mean_f1": None, "std_f1": None, "n_folds": 0})
                      for col in LABEL_COLS},
        "never_evaluable": never_evaluable,
    }

    agg_path = os.path.join(args.output_dir, "cv", "aggregate.json")
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2, default=str)
    print(f"\n  Aggregate saved to {agg_path}")

    print(f"\n{'='*70}")
    print(f"  PHASE 2 COMPLETE")
    print(f"  Stopping here. Do NOT proceed to Phase 3 without review.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
