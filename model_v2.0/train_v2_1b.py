"""
Model v2.1b Retraining Pipeline:
1. Dynamic UA Augmentation during training:
   - Each epoch, for each sample with p=0.5, replace UA with either "masked" (50%)
     or a random UA drawn from the training pool (50%).
   - Evaluation always uses original UAs.
2. 3 Epochs Max:
   - Evaluates full validation set after each epoch.
   - Prints Val Loss, Val Macro-F1 (All 13), and Val Macro-F1 (Excl. 272).
   - Keeps best weights by Val Macro-F1 (Excl. 272), saves every epoch checkpoint.
3. Post-Training Evaluation:
   - Tunes per-class thresholds on validation set.
   - Dual test evaluation: Row-weighted test & Unique-template test.
   - Side-by-side reporting: Val vs Test (Row-Weighted) vs Test (Unique-Template) vs Combined (Val+Test).
   - FPR tables (with and without 272) across UA groups and Date buckets.
   - Rerun UA-swap robustness test across 5 common browser UAs.
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.parse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

# ── Config & Constants ────────────────────────────────────────────────
BASE_MODEL = "distilbert-base-uncased"
MAX_LENGTH = 256
BATCH_SIZE = 64
LEARNING_RATE = 2e-5
DROPOUT = 0.3
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
SIGMOID_THRESHOLD = 0.5
MAX_BODY_CHARS = 500
RANDOM_SEED = 42

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

SWAP_UAS = [
    ("Firefox 73 (Linux)", "Mozilla/5.0 (X11; Linux x86_64; rv:73.0) Gecko/20100101 Firefox/73.0"),
    ("MSIE 7.0 (Win 7)", "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.1; Trident/6.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; Media Center PC 6.0; .NET4.0C; .NET4.0E)"),
    ("Chromium 37 (Ubuntu)", "Mozilla/5.0 (X11; Linux i686) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/37.0.2062.120 Chrome/37.0.2062.120 Safari/537.36"),
    ("Safari 8 (Mac OS X)", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/600.2.5 (KHTML, like Gecko) Version/8.0 Safari/600.1.25"),
    ("IE 11 (Win 8.1)", "Mozilla/5.0 (Windows NT 6.3; Win64; x64; Trident/7.0; Touch; MAARJS; rv:11.0) like Gecko"),
]


# ── Model & Dataset ───────────────────────────────────────────────────
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


class WAFDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=MAX_LENGTH):
        batch_size = 4096
        all_ids, all_masks = [], []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            enc = tokenizer(chunk, truncation=True, max_length=max_length,
                            padding="max_length", return_tensors="pt")
            all_ids.append(enc["input_ids"])
            all_masks.append(enc["attention_mask"])
        self.input_ids = torch.cat(all_ids, dim=0)
        self.attention_mask = torch.cat(all_masks, dim=0)
        self.labels = torch.tensor(labels, dtype=torch.float32) if labels is not None else None

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item


# ── Serialization Functions ───────────────────────────────────────────
def fast_serialize(df, override_ua=None):
    """Vectorized serialization with optional UA override array."""
    method = df['request_http_method'].fillna('').astype(str).str.strip().replace({'nan': '', 'none': ''})
    path = df['request_http_request'].fillna('').astype(str).str.strip().replace({'nan': '', 'none': ''})
    if override_ua is not None:
        ua = pd.Series(override_ua, index=df.index).astype(str).str.strip()
    else:
        ua = df['request_user_agent'].fillna('').astype(str).str.strip().replace({'nan': '', 'none': ''})
    ct = df['request_content_type'].fillna('').astype(str).str.strip().replace({'nan': '', 'none': ''})
    body = df['request_body'].fillna('').astype(str).str.strip().replace({'nan': '', 'none': ''}).str.slice(0, MAX_BODY_CHARS)

    parts = []
    for tag, s in [('METHOD', method), ('PATH', path), ('UA', ua), ('CT', ct), ('BODY', body)]:
        part = np.where((s != '') & (s.str.lower() != 'nan') & (s.str.lower() != 'none'), f'[{tag}] ' + s, '')
        parts.append(part)

    stacked = np.column_stack(parts)
    res = [' '.join([p for p in row if p]) for row in stacked]
    return [r if r else '[EMPTY]' for r in res]


def augment_training_uas(df_train, all_train_uas, rng):
    """
    Dynamic UA augmentation for training:
    For each sample with p=0.5, keep original UA.
    With p=0.5:
      - 50% chance: replace with "masked"
      - 50% chance: replace with random UA drawn from training pool
    """
    orig_uas = df_train['request_user_agent'].fillna('').astype(str).values
    n = len(orig_uas)

    # Decisions: 0 = keep, 1 = masked, 2 = random_ua
    decisions = rng.choice([0, 1, 2], size=n, p=[0.50, 0.25, 0.25])
    random_pool_picks = rng.choice(all_train_uas, size=n)

    augmented_uas = np.where(decisions == 0, orig_uas,
                             np.where(decisions == 1, 'masked', random_pool_picks))
    return augmented_uas


# ── Evaluation & Metrics ──────────────────────────────────────────────
def score_texts(model, tokenizer, texts, device, desc="Scoring"):
    model.eval()
    net = nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
    ds = WAFDataset(texts, None, tokenizer, MAX_LENGTH)
    loader = DataLoader(ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2, pin_memory=True)

    all_probs = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, bar_format='{l_bar}{bar:30}{r_bar}'):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                outputs = net(input_ids, attention_mask)
            probs = torch.sigmoid(outputs["logits"]).cpu().numpy()
            all_probs.append(probs)

    del ds, loader
    torch.cuda.empty_cache()
    return np.concatenate(all_probs, axis=0)


def compute_metrics(labels, probs, thresholds):
    results = {}
    f1s = []
    preds = np.zeros_like(probs, dtype=int)
    for j, col in enumerate(LABEL_COLS):
        t = thresholds[j] if isinstance(thresholds, (list, np.ndarray)) else thresholds
        preds[:, j] = (probs[:, j] >= t).astype(int)

        p = float(precision_score(labels[:, j], preds[:, j], zero_division=0))
        r = float(recall_score(labels[:, j], preds[:, j], zero_division=0))
        f = float(f1_score(labels[:, j], preds[:, j], zero_division=0))
        sup = int(labels[:, j].sum())
        results[col] = {"precision": p, "recall": r, "f1": f, "support": sup, "threshold": float(t)}
        f1s.append(f)

    macro_all = float(np.mean(f1s))
    # Exclude 272 (index 1)
    f1s_no_272 = [f1s[j] for j in range(len(LABEL_COLS)) if j != 1]
    macro_no_272 = float(np.mean(f1s_no_272))
    micro_f1 = float(f1_score(labels, preds, average="micro", zero_division=0))

    return {
        "per_class": results,
        "macro_f1_all": macro_all,
        "macro_f1_no_272": macro_no_272,
        "micro_f1": micro_f1,
        "preds": preds
    }


def tune_val_thresholds(val_labels, val_probs):
    optimal_thresholds = []
    for j in range(len(LABEL_COLS)):
        best_t = 0.5
        best_f1 = f1_score(val_labels[:, j], (val_probs[:, j] >= 0.5).astype(int), zero_division=0)
        for t in np.arange(0.10, 0.92, 0.05):
            f = f1_score(val_labels[:, j], (val_probs[:, j] >= t).astype(int), zero_division=0)
            if f > best_f1:
                best_f1 = f
                best_t = float(t)
        optimal_thresholds.append(best_t)
    return np.array(optimal_thresholds)


# ── Main Training Loop ────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Model v2.1b Retraining Pipeline")
    parser.add_argument("--data_path", default="data/data_capec_multilabel.csv",
                        help="Path to data_capec_multilabel.csv")
    parser.add_argument("--splits_dir", default="outputs/splits",
                        help="Path to precomputed split indices")
    parser.add_argument("--output_dir", default="outputs/v2_1b",
                        help="Output directory for checkpoints & metrics")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs (default: 3)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, Available GPUs: {torch.cuda.device_count()}")

    # 1. Load Data
    print(f"\nLoading dataset from {args.data_path}...")
    df = pd.read_csv(args.data_path, low_memory=False)
    if '248 - Command Injection' in df.columns and '88 - OS Command Injection' in df.columns:
        df['88 - OS Command Injection'] = (
            df['88 - OS Command Injection'].fillna(0).astype(int) |
            df['248 - Command Injection'].fillna(0).astype(int)
        ).astype(int)
        df = df.drop(columns=['248 - Command Injection'])

    df = df.dropna(subset=['request_http_request']).reset_index(drop=True)
    labels_np = df[LABEL_COLS].fillna(0).astype(int).values
    print(f"Loaded {len(df):,d} cleaned rows.")

    # 2. Load Split Indices
    splits_dir = args.splits_dir
    tr_path = os.path.join(splits_dir, "grouped_train_final_idx.npy")
    if not os.path.exists(tr_path):
        for candidate_zip in [
            os.path.join("outputs", "splits_phase_c.zip"),
            os.path.join(os.path.dirname(splits_dir), "splits_phase_c.zip"),
            "splits_phase_c.zip"
        ]:
            if os.path.exists(candidate_zip):
                import zipfile
                print(f"[INFO] Extracting split archive from {candidate_zip} to {splits_dir}...")
                os.makedirs(splits_dir, exist_ok=True)
                with zipfile.ZipFile(candidate_zip, 'r') as z:
                    z.extractall(splits_dir)
                break

    final_tr_idx = np.load(os.path.join(splits_dir, "grouped_train_final_idx.npy"))
    grouped_val_idx = np.load(os.path.join(splits_dir, "grouped_val_idx.npy"))
    grouped_test_rw_idx = np.load(os.path.join(splits_dir, "grouped_test_idx.npy"))
    grouped_test_ut_idx = np.load(os.path.join(splits_dir, "grouped_test_unique_template_idx.npy"))

    df_train = df.iloc[final_tr_idx].copy().reset_index(drop=True)
    df_val = df.iloc[grouped_val_idx].copy().reset_index(drop=True)
    df_test_rw = df.iloc[grouped_test_rw_idx].copy().reset_index(drop=True)
    df_test_ut = df.iloc[grouped_test_ut_idx].copy().reset_index(drop=True)

    tr_labels = labels_np[final_tr_idx]
    val_labels = labels_np[grouped_val_idx]
    test_rw_labels = labels_np[grouped_test_rw_idx]
    test_ut_labels = labels_np[grouped_test_ut_idx]

    print(f"Split sizes: Train={len(df_train):,d}, Val={len(df_val):,d}, Test(RW)={len(df_test_rw):,d}, Test(UT)={len(df_test_ut):,d}")

    # Clipped pos_weights
    pos_counts = tr_labels.sum(axis=0)
    raw_pos_weights = (len(tr_labels) - pos_counts) / np.maximum(pos_counts, 1)
    clipped_pos_weights = np.clip(raw_pos_weights, 1.0, 50.0)
    pos_weights = torch.tensor(clipped_pos_weights, dtype=torch.float32).to(device)

    # Pool of UAs in the training set for random sampling
    all_train_uas = df_train['request_user_agent'].fillna('').astype(str).tolist()

    # Pre-serialize validation set (always original UAs)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    print("\nPre-serializing validation set...")
    val_texts = fast_serialize(df_val, override_ua=None)
    val_ds = WAFDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2, pin_memory=True)

    # Initialize model
    raw_model = MultiLabelWAFModel().to(device)
    model = nn.DataParallel(raw_model) if torch.cuda.device_count() > 1 else raw_model

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # 3 epochs
    epochs = args.epochs
    total_steps = (len(df_train) // BATCH_SIZE) * epochs
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.cuda.amp.GradScaler()

    best_val_macro_no_272 = -1.0
    best_epoch = -1
    best_weights_path = os.path.join(args.output_dir, "v2_1b_best.pt")
    rng = np.random.RandomState(RANDOM_SEED)

    print("\n" + "="*80)
    print(f"TRAINING v2.1b (3 EPOCHS MAX, DYNAMIC UA AUGMENTATION p=0.5)")
    print("="*80)

    for epoch in range(epochs):
        epoch_start = time.time()
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")

        # 1. Dynamic UA Augmentation for this epoch
        t_aug = time.time()
        augmented_uas = augment_training_uas(df_train, all_train_uas, rng)
        n_masked = (augmented_uas == 'masked').sum()
        n_swapped = ((augmented_uas != 'masked') & (augmented_uas != df_train['request_user_agent'].fillna('').values)).sum()
        print(f"  UA Augmentation: {n_masked:,d} masked ({100.0*n_masked/len(df_train):.1f}%), {n_swapped:,d} random UA swaps ({100.0*n_swapped/len(df_train):.1f}%)")

        train_texts = fast_serialize(df_train, override_ua=augmented_uas)
        train_ds = WAFDataset(train_texts, tr_labels, tokenizer, MAX_LENGTH)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

        # 2. Train Loop
        model.train()
        train_loss = 0.0
        progress = tqdm(train_loader, desc=f"  v2.1b Epoch {epoch+1}/{epochs}", bar_format='{l_bar}{bar:30}{r_bar}')

        for step, batch in enumerate(progress):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                outputs = model(input_ids, attention_mask)
                loss = loss_fn(outputs["logits"], labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            train_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}", avg=f"{train_loss/(step+1):.4f}")

        # 3. Validation Evaluation (Original UAs)
        model.eval()
        val_loss = 0.0
        val_probs_epoch = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids, attention_mask)
                    loss = loss_fn(outputs["logits"], labels)
                val_loss += loss.item()
                val_probs_epoch.append(torch.sigmoid(outputs["logits"]).cpu().numpy())

        val_loss /= len(val_loader)
        val_probs_epoch = np.concatenate(val_probs_epoch, axis=0)

        # Compute metrics at default 0.50
        epoch_metrics = compute_metrics(val_labels, val_probs_epoch, 0.5)
        m_all = epoch_metrics["macro_f1_all"]
        m_no_272 = epoch_metrics["macro_f1_no_272"]
        epoch_time = time.time() - epoch_start

        print(f"  Epoch {epoch+1} finished in {epoch_time:.0f}s | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f}")
        print(f"  Validation Macro-F1 (All 13):        {m_all:.4f}")
        print(f"  Validation Macro-F1 (Excluding 272):  {m_no_272:.4f}")

        # Save epoch checkpoint
        epoch_ckpt = os.path.join(args.output_dir, f"v2_1b_epoch_{epoch+1}.pt")
        torch.save(raw_model.state_dict(), epoch_ckpt)
        print(f"  Saved epoch checkpoint to {epoch_ckpt}")

        # Best epoch tracking
        if m_no_272 > best_val_macro_no_272:
            best_val_macro_no_272 = m_no_272
            best_epoch = epoch + 1
            torch.save(raw_model.state_dict(), best_weights_path)
            print(f"  [BEST] New best model saved! (Val Macro-F1 Excl. 272: {best_val_macro_no_272:.4f})")

        del train_ds, train_loader
        torch.cuda.empty_cache()

    print("\n" + "="*80)
    print(f"TRAINING COMPLETE. Best Epoch: {best_epoch} (Val Macro-F1 Excl. 272 = {best_val_macro_no_272:.4f})")
    print(f"Loading best weights from {best_weights_path} for final evaluation...")
    print("="*80)

    raw_model.load_state_dict(torch.load(best_weights_path, map_location=device))
    raw_model.eval()

    # 4. Final Evaluation & Threshold Tuning
    print("\nScoring full validation set with best model...")
    val_probs = score_texts(raw_model, tokenizer, val_texts, device, desc="Val Scoring")
    tuned_thresholds = tune_val_thresholds(val_labels, val_probs)

    print("\nScoring Row-Weighted Test Set...")
    test_rw_texts = fast_serialize(df_test_rw, override_ua=None)
    test_rw_probs = score_texts(raw_model, tokenizer, test_rw_texts, device, desc="Test RW Scoring")

    print("\nScoring Unique-Template Test Set...")
    test_ut_texts = fast_serialize(df_test_ut, override_ua=None)
    test_ut_probs = score_texts(raw_model, tokenizer, test_ut_texts, device, desc="Test UT Scoring")

    # Combined Val + Test
    comb_labels = np.concatenate([val_labels, test_rw_labels], axis=0)
    comb_probs = np.concatenate([val_probs, test_rw_probs], axis=0)

    # Compute metrics
    val_def = compute_metrics(val_labels, val_probs, 0.5)
    val_tun = compute_metrics(val_labels, val_probs, tuned_thresholds)

    rw_def = compute_metrics(test_rw_labels, test_rw_probs, 0.5)
    rw_tun = compute_metrics(test_rw_labels, test_rw_probs, tuned_thresholds)

    ut_def = compute_metrics(test_ut_labels, test_ut_probs, 0.5)
    ut_tun = compute_metrics(test_ut_labels, test_ut_probs, tuned_thresholds)

    comb_def = compute_metrics(comb_labels, comb_probs, 0.5)
    comb_tun = compute_metrics(comb_labels, comb_probs, tuned_thresholds)

    # 5. Side-by-Side Comparison Table (Val vs Test RW vs Test UT vs Combined)
    print("\n\n" + "="*120)
    print("MODEL v2.1b FINAL SIDE-BY-SIDE EVALUATION COMPARISON (F1 SCORES)")
    print("="*120)
    print(f"{'Class':<38s} | {'VAL (Tuned)':>12s} | {'TEST RW (Def)':>14s} {'TEST RW (Tun)':>14s} | {'TEST UT (Tun)':>14s} | {'COMBINED (Tun)':>15s}")
    print("-" * 120)

    for col in LABEL_COLS:
        v_f = val_tun['per_class'][col]['f1']
        rw_d = rw_def['per_class'][col]['f1']
        rw_t = rw_tun['per_class'][col]['f1']
        ut_t = ut_tun['per_class'][col]['f1']
        cb_t = comb_tun['per_class'][col]['f1']
        print(f"{col:<38s} | {v_f:>12.3f} | {rw_d:>14.3f} {rw_t:>14.3f} | {ut_t:>14.3f} | {cb_t:>15.3f}")

    print("-" * 120)
    print(f"{'Macro-F1 (All 13 Classes)':<38s} | {val_tun['macro_f1_all']:>12.3f} | {rw_def['macro_f1_all']:>14.3f} {rw_tun['macro_f1_all']:>14.3f} | {ut_tun['macro_f1_all']:>14.3f} | {comb_tun['macro_f1_all']:>15.3f}")
    print(f"{'Macro-F1 (EXCLUDING 272)':<38s} | {val_tun['macro_f1_no_272']:>12.3f} | {rw_def['macro_f1_no_272']:>14.3f} {rw_tun['macro_f1_no_272']:>14.3f} | {ut_tun['macro_f1_no_272']:>14.3f} | {comb_tun['macro_f1_no_272']:>15.3f}")
    print("="*120)

    # 6. FPR Tables on Normal Test Rows (With and Without 272)
    print("\n\n" + "="*95)
    print("FALSE-POSITIVE ANALYSIS ON TRUE-NORMAL TEST ROWS (v2.1b)")
    print("="*95)

    is_true_normal = (test_rw_labels[:, 0] == 1) & (test_rw_labels[:, 1:].sum(axis=1) == 0)
    norm_indices = np.where(is_true_normal)[0]
    df_normal = df_test_rw.iloc[norm_indices].copy()
    norm_probs = test_rw_probs[norm_indices]

    df_normal['is_fp_all'] = (norm_probs[:, 1:] >= SIGMOID_THRESHOLD).any(axis=1)
    attack_indices_no_272 = [j for j in range(1, 13) if j != 1]
    df_normal['is_fp_no_272'] = (norm_probs[:, attack_indices_no_272] >= SIGMOID_THRESHOLD).any(axis=1)

    # (i) UA-Group
    ff70_ua = 'Mozilla/5.0 (X11; Linux x86_64; rv:70.0) Gecko/20100101 Firefox/70.0'
    is_ff70 = df_normal['request_user_agent'] == ff70_ua

    ff_cnt = int(is_ff70.sum())
    ff_all = int(df_normal.loc[is_ff70, 'is_fp_all'].sum())
    ff_no = int(df_normal.loc[is_ff70, 'is_fp_no_272'].sum())

    other_cnt = int((~is_ff70).sum())
    other_all = int(df_normal.loc[~is_ff70, 'is_fp_all'].sum())
    other_no = int(df_normal.loc[~is_ff70, 'is_fp_no_272'].sum())

    print(f"\n(i) False Positive Rate by User-Agent Group:")
    print(f"{'Group':<32s} {'Normal Rows':>12s} | {'FPs (All)':>10s} {'FPR (All)':>10s} | {'FPs (No 272)':>12s} {'FPR (No 272)':>12s}")
    print("-" * 88)
    print(f"{'Firefox 70 (Campaign UA)':<32s} {ff_cnt:>12,d} | {ff_all:>10,d} {100.0*ff_all/ff_cnt:>9.2f}% | {ff_no:>12,d} {100.0*ff_no/ff_cnt:>11.2f}%")
    print(f"{'All Other User-Agents':<32s} {other_cnt:>12,d} | {other_all:>10,d} {100.0*other_all/other_cnt:>9.2f}% | {other_no:>12,d} {100.0*other_no/other_cnt:>11.2f}%")

    # (ii) Date-Bucket
    ts_test = pd.to_datetime(df_normal['timestamp'], format="%d/%b/%Y:%H:%M:%S %z", errors='coerce', utc=True)
    date_test = ts_test.dt.floor('D')

    def get_bucket(d):
        if pd.isna(d):
            return 'Other'
        d_str = str(d).split(' ')[0]
        if '2020-07-17' <= d_str <= '2020-07-19':
            return '2020-07-17..19 (Campaign)'
        elif '2020-07-20' <= d_str <= '2020-07-28':
            return '2020-07-20..28 (Post-Campaign)'
        elif d_str == '2020-07-29':
            return '2020-07-29 (Surge Day)'
        else:
            return 'Other'

    df_normal['date_bucket'] = date_test.apply(get_bucket)

    print(f"\n(ii) False Positive Rate by Calendar Date Bucket:")
    print(f"{'Date Bucket':<32s} {'Normal Rows':>12s} | {'FPs (All)':>10s} {'FPR (All)':>10s} | {'FPs (No 272)':>12s} {'FPR (No 272)':>12s}")
    print("-" * 88)
    for b_name in ['2020-07-17..19 (Campaign)', '2020-07-20..28 (Post-Campaign)', '2020-07-29 (Surge Day)', 'Other']:
        sub_b = df_normal[df_normal['date_bucket'] == b_name]
        b_cnt = len(sub_b)
        b_all = int(sub_b['is_fp_all'].sum())
        b_no = int(sub_b['is_fp_no_272'].sum())
        print(f"{b_name:<32s} {b_cnt:>12,d} | {b_all:>10,d} {100.0*b_all/b_cnt:>9.2f}% | {b_no:>12,d} {100.0*b_no/b_cnt:>11.2f}%")

    # 7. UA-Swap Robustness Test
    print(f"\n\n{'='*95}")
    print("UA-SWAP ROBUSTNESS TEST (v2.1b)")
    print(f"{'='*95}")
    print(f"{'Class':<40s} {'Thresh':>7s} {'Orig TP':>8s} | {'Mean Swap Recall':>16s} | {'Swap Delta':>10s}")
    print("-" * 86)

    for j, col in enumerate(LABEL_COLS):
        thresh = tuned_thresholds[j]
        tp_mask = (test_rw_labels[:, j] == 1) & (test_rw_probs[:, j] >= thresh)
        orig_tp = int(tp_mask.sum())

        if orig_tp == 0:
            print(f"{col:<40s} {thresh:>7.2f} {orig_tp:>8,d} | {'n/a':>16s} | {'n/a':>10s}")
            continue

        df_tp = df_test_rw.iloc[tp_mask].copy()

        swapped_recalls = []
        for name_ua, val_ua in SWAP_UAS:
            df_swapped = df_tp.copy()
            df_swapped['request_user_agent'] = val_ua
            swapped_texts = fast_serialize(df_swapped, override_ua=None)
            probs_swapped = score_texts(raw_model, tokenizer, swapped_texts, device, desc=f"Swap {col[:15]}")
            detected = int((probs_swapped[:, j] >= thresh).sum())
            swapped_recalls.append(detected / orig_tp)

        mean_swap_recall = float(np.mean(swapped_recalls))
        delta_swap = (mean_swap_recall - 1.0) * 100.0
        print(f"{col:<40s} {thresh:>7.2f} {orig_tp:>8,d} | {mean_swap_recall*100:>15.2f}% | {delta_swap:>9.2f}%")

    print("-" * 86)

    # Save summary json (strip non-serializable ndarrays)
    summary_path = os.path.join(args.output_dir, "v2_1b_summary.json")
    def clean_metrics(m):
        return {k: v for k, v in m.items() if k != "preds"}

    with open(summary_path, "w") as f:
        json.dump({
            "best_epoch": best_epoch,
            "best_val_macro_no_272": best_val_macro_no_272,
            "val_metrics": clean_metrics(val_tun),
            "test_rw_metrics": clean_metrics(rw_tun),
            "test_ut_metrics": clean_metrics(ut_tun),
            "combined_metrics": clean_metrics(comb_tun),
        }, f, indent=2)
    print(f"\nSaved v2.1b metrics summary to {summary_path}")


if __name__ == "__main__":
    main()
