"""
Phase C: Template-Disjoint Multi-Label Training & Complete Evaluation Pipeline.

Implements:
1. Split Auto-Discovery & Generation: Automatically discovers split files across
   Kaggle/local directories, or deterministically generates the exact 80/10/10
   template-disjoint & leaky splits on the fly if not found.
2. Integrity Assertions: Verifies disjointness, total row count (907,813), and
   exact match against Phase B train supports (Normal: 420,155; Path Traversal: 16,778; etc.).
3. C2: Deduplicated & capped train split (131,699 samples), pos_weights clipped to [1.0, 50.0].
4. Leaky Baseline: Deduplicated & capped using identical rules, evaluated ONLY on leaky_test_idx.
5. C-extra: Normal model vs UA-masked ablation vs Leaky baseline.
   - Normal model evaluated with both default 0.5 and Val-tuned optimal thresholds.
6. C-extra2: False-positive analysis on Normal test rows (Firefox 70 vs other, calendar date buckets).
7. C-extra3: Protocol Manipulation diagnosis.
8. C-extra4: Residual near-duplicate check on PAYLOAD-ONLY text (PATH + BODY, no framing)
   with Quartile binning (Q1, Q2, Q3, Q4) and per-class F1 per quartile (n/a for <20 rows).
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

# ── Hyperparameters & Config ──────────────────────────────────────────
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

CAPS = {
    '000 - Normal': 30_000,
    '66 - SQL Injection': 30_000,
    '194 - Fake the Source of Data': 20_000,
    '126 - Path Traversal': 15_000,
    '34 - HTTP Response Splitting': 15_000,
}

EXPECTED_PHASE_B_TRAIN = {
    '000 - Normal': 420155,
    '272 - Protocol Manipulation': 7321,
    '242 - Code Injection': 12663,
    '88 - OS Command Injection': 5967,
    '126 - Path Traversal': 16778,
    '66 - SQL Injection': 200267,
    '16 - Dictionary-based Password Attack': 1488,
    '310 - Scanning for Vulnerable Software': 2033,
    '153 - Input Data Manipulation': 1827,
    '274 - HTTP Verb Tampering': 3972,
    '194 - Fake the Source of Data': 44921,
    '34 - HTTP Response Splitting': 15796,
    '33 - HTTP Request Smuggling': 843,
}


# ── Model ─────────────────────────────────────────────────────────────
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


# ── Dataset & DataLoader ──────────────────────────────────────────────
class WAFDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
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
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


# ── Serialization Functions ───────────────────────────────────────────
def fast_serialize(df, mask_ua=False):
    """Vectorized serialization: [METHOD] ... [PATH] ... [UA] ... [CT] ... [BODY]"""
    method = df['request_http_method'].fillna('').astype(str).str.strip().replace({'nan': '', 'none': ''})
    path = df['request_http_request'].fillna('').astype(str).str.strip().replace({'nan': '', 'none': ''})
    if mask_ua:
        ua = pd.Series(['masked'] * len(df), index=df.index)
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


def get_payload_only_texts(df):
    """Payload-only text (PATH + BODY, no METHOD/UA/CT framing)."""
    path = df['request_http_request'].fillna('').astype(str).str.strip()
    body = df['request_body'].fillna('').astype(str).str.strip().str.slice(0, MAX_BODY_CHARS)
    payloads = (path + ' ' + body).str.strip().replace({'': '[EMPTY]'}).values
    return payloads


def check_for_split_archives(splits_dir):
    """Checks for splits_phase_c.zip or splits_phase_c.npz and extracts them if found."""
    search_paths = [
        "splits_phase_c.zip",
        "outputs/splits_phase_c.zip",
        "splits_phase_c.npz",
        "outputs/splits_phase_c.npz",
        "/kaggle/working/splits_phase_c.zip",
        "/kaggle/working/outputs/splits_phase_c.zip",
        "/kaggle/working/splits_phase_c.npz",
        "/kaggle/working/outputs/splits_phase_c.npz",
    ]
    if os.path.exists("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            if "splits_phase_c.zip" in files:
                search_paths.append(os.path.join(root, "splits_phase_c.zip"))
            if "splits_phase_c.npz" in files:
                search_paths.append(os.path.join(root, "splits_phase_c.npz"))

    for p in search_paths:
        if os.path.exists(p):
            if p.endswith(".zip"):
                import zipfile
                print(f"[INFO] Discovered compressed split archive at: {p}")
                os.makedirs(splits_dir, exist_ok=True)
                with zipfile.ZipFile(p, 'r') as z:
                    z.extractall(splits_dir)
                print(f"[INFO] Extracted split files to '{splits_dir}'!\n")
                return True
            elif p.endswith(".npz"):
                print(f"[INFO] Discovered compressed npz archive at: {p}")
                os.makedirs(splits_dir, exist_ok=True)
                data = np.load(p)
                for k in data.files:
                    np.save(os.path.join(splits_dir, f"{k}.npy"), data[k])
                print(f"[INFO] Extracted split files to '{splits_dir}'!\n")
                return True
    return False


# ── Split Auto-Discovery & Generation ─────────────────────────────────
def ensure_splits_exist(splits_dir, df, labels_np):
    """
    Ensures that all required split files exist.
    If precomputed files or zip archives are found in splits_dir or /kaggle/input, they are used.
    If missing, generates exact 80/10/10 template-disjoint & leaky splits on the fly.
    """
    required_files = [
        "grouped_train_idx.npy",
        "grouped_val_idx.npy",
        "grouped_test_idx.npy",
        "grouped_test_unique_template_idx.npy",
        "grouped_train_final_idx.npy",
        "leaky_train_idx.npy",
        "leaky_val_idx.npy",
        "leaky_test_idx.npy",
        "leaky_train_final_idx.npy",
    ]

    # 1. Check specified splits_dir
    if os.path.exists(splits_dir) and all(os.path.exists(os.path.join(splits_dir, f)) for f in required_files):
        return splits_dir

    # 2. Check for zip/npz archives
    if check_for_split_archives(splits_dir):
        if all(os.path.exists(os.path.join(splits_dir, f)) for f in required_files):
            return splits_dir

    # 3. Search common candidate locations (including /kaggle/input and working dir)
    candidate_dirs = [
        splits_dir,
        "outputs/splits",
        "/kaggle/working/outputs/splits",
    ]
    if os.path.exists("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            if "grouped_train_idx.npy" in files:
                candidate_dirs.append(root)

    for c_dir in candidate_dirs:
        if os.path.exists(c_dir) and all(os.path.exists(os.path.join(c_dir, f)) for f in required_files):
            print(f"[INFO] Discovered split files at: {c_dir}")
            return c_dir

    # 4. If missing, auto-generate on the fly
    print("\n" + "="*80)
    print(f"[INFO] Precomputed split files not found in '{splits_dir}' or candidate paths.")
    print("Auto-generating exact 80/10/10 template-disjoint & leaky splits on the fly (seed 42)...")
    print("="*80)

    os.makedirs(splits_dir, exist_ok=True)
    total_rows = len(df)

    # A. Fast Serialize for key building
    print("  1/5 Serializing request strings...")
    serialized = fast_serialize(df, mask_ua=False)

    # B. Build normalized keys
    print("  2/5 Normalizing into template keys...")
    hex_b64_pattern = re.compile(r'\b[0-9a-f]{8,}\b|\b[a-z0-9+/]{16,}={0,2}\b')
    digit_pattern = re.compile(r'\d+')
    space_pattern = re.compile(r'\s+')

    def build_normalized_key(text):
        s = text.lower()
        try:
            s = urllib.parse.unquote(s)
            s = urllib.parse.unquote(s)
        except Exception:
            pass
        s = space_pattern.sub(' ', s).strip()
        s = hex_b64_pattern.sub('H', s)
        s = digit_pattern.sub('0', s)
        return s

    norm_keys = np.array([build_normalized_key(s) for s in serialized])
    df['norm_key'] = norm_keys
    group_ids = pd.factorize(norm_keys)[0]
    dominant_label = labels_np.argmax(axis=1)

    # C. StratifiedGroupKFold 10 splits (seed 42)
    print("  3/5 Computing 80/10/10 StratifiedGroupKFold split (seed 42)...")
    from sklearn.model_selection import StratifiedGroupKFold, train_test_split
    sgkf10 = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
    folds10 = list(sgkf10.split(df, dominant_label, group_ids))

    te_idx = folds10[0][1]
    va_idx = folds10[1][1]
    tr_idx = np.concatenate([folds10[k][1] for k in range(2, 10)])

    # Unique template test
    df_test = df.iloc[te_idx].copy().reset_index()
    unique_test_df = df_test.drop_duplicates(subset=['norm_key'], keep='first')
    unique_template_test_idx = df_test.loc[unique_test_df.index, 'index'].values

    # D. Leaky baseline split (Stratified Random 80/10/10, seed 42)
    print("  4/5 Computing Leaky Stratified Random 80/10/10 split (seed 42)...")
    all_indices = np.arange(total_rows)
    tr_leaky_idx, temp_leaky_idx = train_test_split(
        all_indices, test_size=0.20, random_state=42, stratify=dominant_label
    )
    va_leaky_idx, te_leaky_idx = train_test_split(
        temp_leaky_idx, test_size=0.50, random_state=42, stratify=dominant_label[temp_leaky_idx]
    )

    # E. Deduplicate and cap train splits
    print("  5/5 Deduplicating & capping Grouped and Leaky train splits...")
    def dedupe_and_cap(raw_indices):
        label_tuples = [tuple(row) for row in labels_np[raw_indices]]
        df_tr = pd.DataFrame({'orig_idx': raw_indices, 'norm_key': norm_keys[raw_indices], 'label_tuple': label_tuples})
        rng = np.random.RandomState(42)
        shuffled_order = rng.permutation(len(df_tr))
        df_tr_shuffled = df_tr.iloc[shuffled_order]
        df_dedup = df_tr_shuffled.groupby(['norm_key', 'label_tuple'], as_index=False).head(3)
        dedup_idx = df_dedup['orig_idx'].values

        dom_dedup = labels_np[dedup_idx].argmax(axis=1)
        keep_mask = np.ones(len(dedup_idx), dtype=bool)
        for cls_name, cap in CAPS.items():
            cls_idx = LABEL_COLS.index(cls_name)
            is_cls = dom_dedup == cls_idx
            cls_count = is_cls.sum()
            if cls_count > cap:
                cls_pos = np.where(is_cls)[0]
                drop_pos = rng.choice(cls_pos, size=cls_count - cap, replace=False)
                keep_mask[drop_pos] = False
        return dedup_idx[keep_mask]

    grouped_train_final_idx = dedupe_and_cap(tr_idx)
    leaky_train_final_idx = dedupe_and_cap(tr_leaky_idx)

    # Save all split files
    np.save(os.path.join(splits_dir, "grouped_train_idx.npy"), tr_idx)
    np.save(os.path.join(splits_dir, "grouped_val_idx.npy"), va_idx)
    np.save(os.path.join(splits_dir, "grouped_test_idx.npy"), te_idx)
    np.save(os.path.join(splits_dir, "grouped_test_unique_template_idx.npy"), unique_template_test_idx)
    np.save(os.path.join(splits_dir, "grouped_train_final_idx.npy"), grouped_train_final_idx)

    np.save(os.path.join(splits_dir, "leaky_train_idx.npy"), tr_leaky_idx)
    np.save(os.path.join(splits_dir, "leaky_val_idx.npy"), va_leaky_idx)
    np.save(os.path.join(splits_dir, "leaky_test_idx.npy"), te_leaky_idx)
    np.save(os.path.join(splits_dir, "leaky_train_final_idx.npy"), leaky_train_final_idx)

    print(f"  Successfully auto-generated and saved all splits to '{splits_dir}'!\n")
    return splits_dir


# ── Integrity Assertions ──────────────────────────────────────────────
def run_integrity_assertions(df, labels_np, splits_dir, allow_new_splits=False):
    print("\n" + "="*95)
    print("ITEM 1: INTEGRITY ASSERTIONS (SPLIT DISJOINTNESS & PHASE B TRAIN COUNTS)")
    print("="*95)

    tr_idx = np.load(os.path.join(splits_dir, "grouped_train_idx.npy"))
    va_idx = np.load(os.path.join(splits_dir, "grouped_val_idx.npy"))
    te_idx = np.load(os.path.join(splits_dir, "grouped_test_idx.npy"))
    final_tr_idx = np.load(os.path.join(splits_dir, "grouped_train_final_idx.npy"))

    total_expected = 907813
    actual_total = len(tr_idx) + len(va_idx) + len(te_idx)

    # 1. Disjointness & Length checks
    tr_set, va_set, te_set = set(tr_idx), set(va_idx), set(te_idx)
    assert len(tr_set & va_set) == 0, "FAILED: Train and Val index overlap!"
    assert len(tr_set & te_set) == 0, "FAILED: Train and Test index overlap!"
    assert len(va_set & te_set) == 0, "FAILED: Val and Test index overlap!"
    assert actual_total == total_expected, f"FAILED: Total rows {actual_total} != {total_expected}"

    print(f"Total Cleaned Rows: {actual_total:,d} (Train: {len(tr_idx):,d} | Val: {len(va_idx):,d} | Test: {len(te_idx):,d})")
    print(f"Index Set Disjointness: PASSED (0 overlap between Train, Val, and Test)")
    print(f"Total Rows Check:       PASSED ({actual_total:,d} == {total_expected:,d})")

    # 2. Recompute positive counts on raw train and final train
    raw_tr_sup = labels_np[tr_idx].sum(axis=0)
    final_tr_sup = labels_np[final_tr_idx].sum(axis=0)

    print(f"\n{'Class':<40s} {'Phase B Expected':>17s} {'Actual Raw Train':>17s} {'Match?':>8s} {'Final Train':>13s} {'<= Raw?':>8s}")
    print("-" * 105)

    all_matched = True
    exact_match_count = 0
    for j, col in enumerate(LABEL_COLS):
        exp = EXPECTED_PHASE_B_TRAIN[col]
        actual = int(raw_tr_sup[j])
        final_cnt = int(final_tr_sup[j])

        is_exact = (exp == actual)
        if is_exact:
            exact_match_count += 1
            match_str = "PASSED"
        else:
            match_str = "PASSED*" if allow_new_splits else "FAILED"

        le_str = "PASSED" if final_cnt <= actual else "FAILED"

        if (not is_exact and not allow_new_splits) or final_cnt > actual or actual == 0:
            all_matched = False

        print(f"{col:<40s} {exp:>17,d} {actual:>17,d} {match_str:>8s} {final_cnt:>13,d} {le_str:>8s}")

    if not all_matched:
        print("\n[ERROR] Integrity assertion failed! Raw train supports do not match the exact Phase B split.")
        print("This happens when precomputed split files are not uploaded, causing on-the-fly generation")
        print("under Kaggle's scikit-learn version to partition the 300K groups into slightly different folds.")
        print("\nTo resolve:")
        print("  Option A (Recommended): Upload 'splits_phase_c.zip' (5.1 MB) to Kaggle to use the exact approved Phase B split.")
        print("  Option B: Re-run with '--allow_new_splits' to proceed with the on-the-fly generated partition.\n")
        sys.exit(1)

    print("-" * 105)
    if exact_match_count == len(LABEL_COLS):
        print("ALL INTEGRITY ASSERTIONS PASSED! Raw train supports match Phase B exactly.\n")
    else:
        print("NOTICE: Running with on-the-fly generated split (--allow_new_splits enabled).")
        print("Disjointness, coverage, and capping verified across all 13 classes.\n")


# ── Training Loop ─────────────────────────────────────────────────────
def train_model(model_name_tag, train_texts, train_labels, val_texts, val_labels,
                pos_weights, device, output_dir, epochs=4):
    print(f"\n{'='*70}")
    print(f"  Training {model_name_tag} ({len(train_texts):,d} train samples, {epochs} epochs)")
    print(f"{'='*70}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    print("  Pre-tokenizing train set...")
    train_ds = WAFDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    print("  Pre-tokenizing val set...")
    val_ds = WAFDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
                            num_workers=2, pin_memory=True)

    raw_model = MultiLabelWAFModel().to(device)
    if torch.cuda.device_count() > 1:
        print(f"  Using {torch.cuda.device_count()} GPUs (DataParallel)")
        model = nn.DataParallel(raw_model)
    else:
        model = raw_model

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0

        progress = tqdm(train_loader, desc=f"  {model_name_tag} Epoch {epoch+1}/{epochs}",
                        bar_format='{l_bar}{bar:30}{r_bar}')
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

        # Val loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids, attention_mask)
                    loss = loss_fn(outputs["logits"], labels)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        epoch_time = time.time() - epoch_start
        print(f"  Epoch {epoch+1} done in {epoch_time:.0f}s | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f}")

    save_path = os.path.join(output_dir, f"{model_name_tag}_weights.pt")
    torch.save(raw_model.state_dict(), save_path)
    print(f"  Saved weights to {save_path}")

    del train_ds, val_ds, train_loader, val_loader, optimizer, scheduler, scaler
    torch.cuda.empty_cache()
    return raw_model, tokenizer


# ── Inference & Metrics ───────────────────────────────────────────────
def evaluate_predictions(raw_model, tokenizer, texts, labels, device, desc="Evaluating"):
    raw_model.eval()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(raw_model)
    else:
        model = raw_model

    ds = WAFDataset(texts, labels, tokenizer, MAX_LENGTH)
    loader = DataLoader(ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2, pin_memory=True)

    all_probs = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, bar_format='{l_bar}{bar:30}{r_bar}'):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                outputs = model(input_ids, attention_mask)
            probs = torch.sigmoid(outputs["logits"]).cpu().numpy()
            all_probs.append(probs)

    all_probs = np.concatenate(all_probs, axis=0)
    del ds, loader
    torch.cuda.empty_cache()
    return all_probs


def compute_metrics_with_thresholds(labels, probs, thresholds):
    """Compute precision, recall, and F1 given per-class thresholds."""
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

    macro_f1 = float(np.mean(f1s))
    micro_f1 = float(f1_score(labels, preds, average="micro", zero_division=0))
    return {"per_class": results, "macro_f1": macro_f1, "micro_f1": micro_f1, "preds": preds}


def tune_val_thresholds(val_labels, val_probs):
    """
    Search per-class thresholds in [0.1, 0.9] on val set to maximize F1.
    Tuning on val is crucial because train prevalence differs from test/val
    prevalence due to capping.
    """
    optimal_thresholds = []
    val_f1_default = []
    val_f1_tuned = []

    print("\n" + "="*80)
    print("ITEM 5: VAL-SET THRESHOLD TUNING (0.10 - 0.90 SEARCH)")
    print("="*80)
    print(f"{'Class':<40s} {'Val Support':>12s} {'Def 0.5 F1':>12s} {'Tuned Thresh':>14s} {'Tuned F1':>10s}")
    print("-" * 90)

    for j, col in enumerate(LABEL_COLS):
        best_t = 0.5
        best_f1 = f1_score(val_labels[:, j], (val_probs[:, j] >= 0.5).astype(int), zero_division=0)
        def_f1 = best_f1

        for t in np.arange(0.10, 0.92, 0.05):
            preds_t = (val_probs[:, j] >= t).astype(int)
            f = f1_score(val_labels[:, j], preds_t, zero_division=0)
            if f > best_f1:
                best_f1 = f
                best_t = float(t)

        optimal_thresholds.append(best_t)
        val_f1_default.append(def_f1)
        val_f1_tuned.append(best_f1)

        sup = int(val_labels[:, j].sum())
        print(f"{col:<40s} {sup:>12,d} {def_f1:>12.3f} {best_t:>14.2f} {best_f1:>10.3f}")

    print("-" * 90)
    print(f"{'Macro-F1 on Validation Set':<40s} {'':>12s} {np.mean(val_f1_default):>12.3f} {'':>14s} {np.mean(val_f1_tuned):>10.3f}")
    return np.array(optimal_thresholds)


# ── Main Script ───────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Phase C Training & Evaluation Pipeline")
    parser.add_argument("--data_path", default="data/data_capec_multilabel.csv",
                        help="Path to data_capec_multilabel.csv")
    parser.add_argument("--splits_dir", default="outputs/splits",
                        help="Path to precomputed split indices")
    parser.add_argument("--output_dir", default="outputs/phase_c",
                        help="Output directory for checkpoints & metrics")
    parser.add_argument("--epochs", type=int, default=4,
                        help="Number of training epochs (default: 4)")
    parser.add_argument("--mode", default="all",
                        choices=["all", "normal", "ua_masked", "leaky", "eval_only"],
                        help="Which models to train")
    parser.add_argument("--allow_new_splits", action="store_true", default=False,
                        help="Allow proceeding with on-the-fly generated splits if scikit-learn version produces slightly different folds.")
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
    total_rows = len(df)
    labels_np = df[LABEL_COLS].fillna(0).astype(int).values
    print(f"Loaded {total_rows:,d} cleaned rows.")

    # 2. Ensure Splits Exist (auto-discovers or generates on the fly)
    splits_dir = ensure_splits_exist(args.splits_dir, df, labels_np)

    # 3. Run Item 1: Integrity Assertions
    run_integrity_assertions(df, labels_np, splits_dir, allow_new_splits=args.allow_new_splits)

    # 4. Load Split Indices
    final_tr_idx = np.load(os.path.join(splits_dir, "grouped_train_final_idx.npy"))
    grouped_val_idx = np.load(os.path.join(splits_dir, "grouped_val_idx.npy"))
    grouped_test_rw_idx = np.load(os.path.join(splits_dir, "grouped_test_idx.npy"))
    grouped_test_ut_idx = np.load(os.path.join(splits_dir, "grouped_test_unique_template_idx.npy"))

    final_leaky_tr_idx = np.load(os.path.join(splits_dir, "leaky_train_final_idx.npy"))
    leaky_te_idx = np.load(os.path.join(splits_dir, "leaky_test_idx.npy"))

    # 5. Item 2: pos_weight Calculation & Clipping to [1.0, 50.0]
    print("\n" + "="*80)
    print("ITEM 2: POS_WEIGHT CALCULATION & CLIPPING [1.0, 50.0]")
    print("="*80)
    n_final = len(final_tr_idx)
    pos_counts = labels_np[final_tr_idx].sum(axis=0)
    raw_pos_weights = (n_final - pos_counts) / np.maximum(pos_counts, 1)
    clipped_pos_weights = np.clip(raw_pos_weights, 1.0, 50.0)
    pos_weights = torch.tensor(clipped_pos_weights, dtype=torch.float32)

    print(f"{'Class':<40s} {'Final Train Pos':>15s} {'Raw pos_weight':>15s} {'Clipped [1, 50]':>15s}")
    print("-" * 88)
    for j, col in enumerate(LABEL_COLS):
        print(f"{col:<40s} {pos_counts[j]:>15,d} {raw_pos_weights[j]:>15.2f} {clipped_pos_weights[j]:>15.2f}")

    # Leaky pos weights clipped
    leaky_pos = labels_np[final_leaky_tr_idx].sum(axis=0)
    raw_leaky_pw = (len(final_leaky_tr_idx) - leaky_pos) / np.maximum(leaky_pos, 1)
    leaky_pw = torch.tensor(np.clip(raw_leaky_pw, 1.0, 50.0), dtype=torch.float32)

    # 6. Serialization
    print("\nSerializing request strings...")
    raw_texts = fast_serialize(df, mask_ua=False)
    masked_texts = fast_serialize(df, mask_ua=True)
    payload_only_texts = get_payload_only_texts(df)

    # Label arrays
    tr_labels = labels_np[final_tr_idx]
    val_labels = labels_np[grouped_val_idx]
    test_rw_labels = labels_np[grouped_test_rw_idx]
    test_ut_labels = labels_np[grouped_test_ut_idx]
    leaky_te_labels = labels_np[leaky_te_idx]

    models_to_run = []
    if args.mode in ["all", "normal"]:
        models_to_run.append("normal")
    if args.mode in ["all", "ua_masked"]:
        models_to_run.append("ua_masked")
    if args.mode in ["all", "leaky"]:
        models_to_run.append("leaky")

    all_eval_results = {}
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tuned_thresholds = None

    # 7. Train / Load & Evaluate Models
    for m_type in models_to_run:
        weight_path = os.path.join(args.output_dir, f"{m_type}_model_weights.pt")

        if args.mode != "eval_only":
            if m_type == "normal":
                tr_txt = [raw_texts[i] for i in final_tr_idx]
                va_txt = [raw_texts[i] for i in grouped_val_idx]
                model, tok = train_model("normal_model", tr_txt, tr_labels, va_txt, val_labels,
                                         pos_weights, device, args.output_dir, epochs=args.epochs)
            elif m_type == "ua_masked":
                tr_txt = [masked_texts[i] for i in final_tr_idx]
                va_txt = [masked_texts[i] for i in grouped_val_idx]
                model, tok = train_model("ua_masked_model", tr_txt, tr_labels, va_txt, val_labels,
                                         pos_weights, device, args.output_dir, epochs=args.epochs)
            elif m_type == "leaky":
                tr_txt = [raw_texts[i] for i in final_leaky_tr_idx]
                va_txt = [raw_texts[i] for i in grouped_val_idx]
                model, tok = train_model("leaky_model", tr_txt, labels_np[final_leaky_tr_idx], va_txt, val_labels,
                                         leaky_pw, device, args.output_dir, epochs=args.epochs)
        else:
            print(f"\nLoading weights for {m_type} from {weight_path}...")
            model = MultiLabelWAFModel().to(device)
            model.load_state_dict(torch.load(weight_path, map_location=device))

        # Evaluate Normal Model
        if m_type == "normal":
            # A. Evaluate on Val Set & Tune Thresholds
            va_txt = [raw_texts[i] for i in grouped_val_idx]
            val_probs = evaluate_predictions(model, tokenizer, va_txt, val_labels, device, desc="Normal [Val Probabilities]")
            tuned_thresholds = tune_val_thresholds(val_labels, val_probs)

            # B. Evaluate on Row-Weighted Grouped Test
            te_rw_txt = [raw_texts[i] for i in grouped_test_rw_idx]
            norm_rw_probs = evaluate_predictions(model, tokenizer, te_rw_txt, test_rw_labels, device, desc="Normal [Row-Weighted]")
            rw_metrics_def = compute_metrics_with_thresholds(test_rw_labels, norm_rw_probs, 0.5)
            rw_metrics_tuned = compute_metrics_with_thresholds(test_rw_labels, norm_rw_probs, tuned_thresholds)

            # C. Evaluate on Unique-Template Grouped Test
            te_ut_txt = [raw_texts[i] for i in grouped_test_ut_idx]
            norm_ut_probs = evaluate_predictions(model, tokenizer, te_ut_txt, test_ut_labels, device, desc="Normal [Unique-Template]")
            ut_metrics_def = compute_metrics_with_thresholds(test_ut_labels, norm_ut_probs, 0.5)
            ut_metrics_tuned = compute_metrics_with_thresholds(test_ut_labels, norm_ut_probs, tuned_thresholds)

            all_eval_results["normal"] = {
                "row_weighted_default": rw_metrics_def,
                "row_weighted_tuned": rw_metrics_tuned,
                "unique_template_default": ut_metrics_def,
                "unique_template_tuned": ut_metrics_tuned,
                "normal_rw_probs": norm_rw_probs,
                "normal_ut_probs": norm_ut_probs,
                "normal_ut_preds_def": ut_metrics_def["preds"],
            }

        # Evaluate UA-Masked Model
        elif m_type == "ua_masked":
            te_rw_txt = [masked_texts[i] for i in grouped_test_rw_idx]
            te_ut_txt = [masked_texts[i] for i in grouped_test_ut_idx]

            mask_rw_probs = evaluate_predictions(model, tokenizer, te_rw_txt, test_rw_labels, device, desc="UA-Masked [Row-Weighted]")
            rw_metrics_def = compute_metrics_with_thresholds(test_rw_labels, mask_rw_probs, 0.5)

            mask_ut_probs = evaluate_predictions(model, tokenizer, te_ut_txt, test_ut_labels, device, desc="UA-Masked [Unique-Template]")
            ut_metrics_def = compute_metrics_with_thresholds(test_ut_labels, mask_ut_probs, 0.5)

            all_eval_results["ua_masked"] = {
                "row_weighted_default": rw_metrics_def,
                "unique_template_default": ut_metrics_def,
            }

        # Evaluate Leaky Model (ONLY on leaky_test_idx as specified in Item 3)
        elif m_type == "leaky":
            print(f"\n[ITEM 3] Evaluating Leaky Model ONLY on leaky_test_idx ({len(leaky_te_idx):,d} samples)...")
            leaky_te_txt = [raw_texts[i] for i in leaky_te_idx]
            leaky_probs = evaluate_predictions(model, tokenizer, leaky_te_txt, leaky_te_labels, device, desc="Leaky [Leaky Test]")
            leaky_metrics_def = compute_metrics_with_thresholds(leaky_te_labels, leaky_probs, 0.5)

            all_eval_results["leaky"] = {
                "leaky_test_default": leaky_metrics_def,
            }

    # 8. C-extra2: False-Positive Analysis on Normal Model
    if "normal" in all_eval_results and all_eval_results["normal"]["normal_rw_probs"] is not None:
        print("\n" + "="*80)
        print("C-extra2: FALSE-POSITIVE ANALYSIS (NORMAL MODEL ON ROW-WEIGHTED TEST)")
        print("="*80)
        norm_rw_probs = all_eval_results["normal"]["normal_rw_probs"]
        is_gt_normal = (test_rw_labels[:, 0] == 1) & (test_rw_labels[:, 1:].sum(axis=1) == 0)
        norm_indices_in_test = np.where(is_gt_normal)[0]
        norm_df_test = df.iloc[grouped_test_rw_idx[norm_indices_in_test]].copy()

        attack_preds = (norm_rw_probs[norm_indices_in_test, 1:] >= SIGMOID_THRESHOLD).any(axis=1)
        norm_df_test['is_fp'] = attack_preds

        # (a) UA == Firefox 70 vs Other
        ff70_ua = 'Mozilla/5.0 (X11; Linux x86_64; rv:70.0) Gecko/20100101 Firefox/70.0'
        is_ff70 = norm_df_test['request_user_agent'] == ff70_ua
        ff_cnt = is_ff70.sum()
        ff_fp = norm_df_test.loc[is_ff70, 'is_fp'].sum()
        ff_fpr = 100.0 * ff_fp / ff_cnt if ff_cnt > 0 else 0.0

        other_cnt = (~is_ff70).sum()
        other_fp = norm_df_test.loc[~is_ff70, 'is_fp'].sum()
        other_fpr = 100.0 * other_fp / other_cnt if other_cnt > 0 else 0.0

        print(f"\n(a) False Positive Rate by User-Agent Group:")
        print(f"{'Group':<35s} {'Normal Rows':>12s} {'FP Count':>10s} {'FPR (%)':>10s}")
        print("-" * 70)
        print(f"{'Firefox 70 (Campaign UA)':<35s} {ff_cnt:>12,d} {ff_fp:>10,d} {ff_fpr:>9.2f}%")
        print(f"{'All Other User-Agents':<35s} {other_cnt:>12,d} {other_fp:>10,d} {other_fpr:>9.2f}%")

        if ff_fpr > 2 * other_fpr:
            print(f"  [FLAG] ALERT: Firefox-70 campaign benign rows have {ff_fpr/other_fpr:.1f}x HIGHER FPR than other UAs!")

        # (b) Calendar date bucket
        ts_test = pd.to_datetime(norm_df_test['timestamp'], format="%d/%b/%Y:%H:%M:%S %z", errors='coerce', utc=True)
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

        norm_df_test['date_bucket'] = date_test.apply(get_bucket)
        print(f"\n(b) False Positive Rate by Calendar Date Bucket:")
        print(f"{'Date Bucket':<35s} {'Normal Rows':>12s} {'FP Count':>10s} {'FPR (%)':>10s}")
        print("-" * 70)
        for b_name in ['2020-07-17..19 (Campaign)', '2020-07-20..28 (Post-Campaign)', '2020-07-29 (Surge Day)', 'Other']:
            sub_b = norm_df_test[norm_df_test['date_bucket'] == b_name]
            b_cnt = len(sub_b)
            b_fp = sub_b['is_fp'].sum()
            b_fpr = 100.0 * b_fp / b_cnt if b_cnt > 0 else 0.0
            print(f"{b_name:<35s} {b_cnt:>12,d} {b_fp:>10,d} {b_fpr:>9.2f}%")

    # 9. C-extra4: Residual Near-Duplicate Check on PAYLOAD-ONLY Text with Quartile Bins
    if "normal" in all_eval_results and all_eval_results["normal"]["normal_ut_preds_def"] is not None:
        print("\n" + "="*80)
        print("ITEM 4: C-EXTRA4 RESIDUAL NEAR-DUPLICATE CHECK (PAYLOAD-ONLY & QUARTILES)")
        print("="*80)

        rng = np.random.RandomState(RANDOM_SEED)
        n_sample_ut = min(20000, len(grouped_test_ut_idx))
        sample_ut_indices = rng.choice(len(grouped_test_ut_idx), size=n_sample_ut, replace=False)

        # PAYLOAD-ONLY text (PATH + BODY, no METHOD/UA/CT framing)
        sample_ut_texts = [payload_only_texts[grouped_test_ut_idx[i]] for i in sample_ut_indices]
        sample_ut_labels = test_ut_labels[sample_ut_indices]
        sample_ut_preds = all_eval_results["normal"]["normal_ut_preds_def"][sample_ut_indices]

        # Reference train pool (payload-only text from final train set)
        n_sample_tr = min(200000, len(final_tr_idx))
        sample_tr_indices = rng.choice(final_tr_idx, size=n_sample_tr, replace=False)
        sample_tr_texts = [payload_only_texts[i] for i in sample_tr_indices]

        print(f"Fitting TF-IDF char n-grams (3-5) on {len(sample_tr_texts):,d} training payloads...")
        t_vec = time.time()
        tfidf = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), max_features=25000, sublinear_tf=True)
        X_tr = normalize(tfidf.fit_transform(sample_tr_texts))
        X_te = normalize(tfidf.transform(sample_ut_texts))
        print(f"Vectorized in {time.time()-t_vec:.1f}s. Computing nearest-neighbor cosine similarities...")

        # Fast batched dot product
        batch_size = 2000
        max_sims = []
        for i in range(0, X_te.shape[0], batch_size):
            sim_batch = X_te[i:i+batch_size].dot(X_tr.T).toarray()
            max_sims.append(sim_batch.max(axis=1))
        max_sims = np.concatenate(max_sims)

        min_sim = float(np.min(max_sims))
        q25 = float(np.percentile(max_sims, 25))
        q50 = float(np.median(max_sims))
        q75 = float(np.percentile(max_sims, 75))
        max_sim = float(np.max(max_sims))
        p90_sim = float(np.percentile(max_sims, 90))

        print(f"\nPayload-Only Cosine Similarity Distribution (Test Sample -> Train NN):")
        print(f"  Min Similarity:           {min_sim:.4f}")
        print(f"  Quartile 1 (Q1, 25%):     {q25:.4f}")
        print(f"  Median (Q2, 50%):         {q50:.4f}")
        print(f"  Quartile 3 (Q3, 75%):     {q75:.4f}")
        print(f"  Max Similarity:           {max_sim:.4f}")
        print(f"  90th Percentile:          {p90_sim:.4f}")

        # Define 4 quartile bins
        q_bins = [
            (f"Q1 [<= {q25:.3f}]", max_sims <= q25),
            (f"Q2 ({q25:.3f}, {q50:.3f}]", (max_sims > q25) & (max_sims <= q50)),
            (f"Q3 ({q50:.3f}, {q75:.3f}]", (max_sims > q50) & (max_sims <= q75)),
            (f"Q4 [> {q75:.3f}]", max_sims > q75),
        ]

        print(f"\nPer-Class F1 across Similarity Quartiles (mark n/a if positive test rows < 20):")
        header_bins = "   ".join([f"{b[0]:^18s}" for b in q_bins])
        print(f"{'Class':<38s}   {header_bins}")
        print("-" * 118)

        for j, col in enumerate(LABEL_COLS):
            q_cols = []
            for b_name, b_mask in q_bins:
                n_pos = int(sample_ut_labels[b_mask, j].sum())
                if n_pos < 20:
                    q_cols.append(f"{'n/a (N=' + str(n_pos) + ')':^18s}")
                else:
                    score = f1_score(sample_ut_labels[b_mask, j], sample_ut_preds[b_mask, j], zero_division=0)
                    cell_text = f"{score:.3f} (N={n_pos})"
                    q_cols.append(f"{cell_text:^18s}")
            print(f"{col:<38s}   {'   '.join(q_cols)}")

    # 10. Headline Comparison Table
    print("\n\n" + "="*110)
    print("FINAL HEADLINE EVALUATION COMPARISON (F1 SCORES)")
    print("="*110)
    print(f"{'Class':<36s} | {'--- GROUPED ROW-WEIGHTED ---':^28s} | {'-- GROUPED UNIQUE-TEMPLATE --':^28s} | {'LEAKY TEST':^11s}")
    print(f"{'':<36s} | {'Normal(0.5)':>11s} {'Normal(Tun)':>11s} {'UA-Mask':>7s} | {'Normal(0.5)':>11s} {'Normal(Tun)':>11s} {'UA-Mask':>7s} | {'Leaky(0.5)':>11s}")
    print("-" * 110)

    for col in LABEL_COLS:
        def get_f1_val(m_dict, col_name):
            if m_dict is not None and col_name in m_dict['per_class']:
                return f"{m_dict['per_class'][col_name]['f1']:>11.3f}"
            return "        n/a"

        def get_f1_short(m_dict, col_name):
            if m_dict is not None and col_name in m_dict['per_class']:
                return f"{m_dict['per_class'][col_name]['f1']:>7.3f}"
            return "    n/a"

        # Grouped Row-weighted
        rw_norm_def = get_f1_val(all_eval_results.get("normal", {}).get("row_weighted_default"), col)
        rw_norm_tun = get_f1_val(all_eval_results.get("normal", {}).get("row_weighted_tuned"), col)
        rw_uamask = get_f1_short(all_eval_results.get("ua_masked", {}).get("row_weighted_default"), col)

        # Grouped Unique-template
        ut_norm_def = get_f1_val(all_eval_results.get("normal", {}).get("unique_template_default"), col)
        ut_norm_tun = get_f1_val(all_eval_results.get("normal", {}).get("unique_template_tuned"), col)
        ut_uamask = get_f1_short(all_eval_results.get("ua_masked", {}).get("unique_template_default"), col)

        # Leaky on leaky test
        leak_f1 = get_f1_val(all_eval_results.get("leaky", {}).get("leaky_test_default"), col)

        print(f"{col:<36s} | {rw_norm_def} {rw_norm_tun} {rw_uamask} | {ut_norm_def} {ut_norm_tun} {ut_uamask} | {leak_f1}")

    print("-" * 110)
    def get_macro_str(m_dict, width=11):
        if m_dict is not None:
            return f"{m_dict['macro_f1']:>{width}.3f}"
        return f"{'n/a':>{width}s}"

    m_rw_def = get_macro_str(all_eval_results.get("normal", {}).get("row_weighted_default"), 11)
    m_rw_tun = get_macro_str(all_eval_results.get("normal", {}).get("row_weighted_tuned"), 11)
    m_rw_msk = get_macro_str(all_eval_results.get("ua_masked", {}).get("row_weighted_default"), 7)

    m_ut_def = get_macro_str(all_eval_results.get("normal", {}).get("unique_template_default"), 11)
    m_ut_tun = get_macro_str(all_eval_results.get("normal", {}).get("unique_template_tuned"), 11)
    m_ut_msk = get_macro_str(all_eval_results.get("ua_masked", {}).get("unique_template_default"), 7)

    m_leak = get_macro_str(all_eval_results.get("leaky", {}).get("leaky_test_default"), 11)

    print(f"{'Macro-F1 (All 13 Classes)':<36s} | {m_rw_def} {m_rw_tun} {m_rw_msk} | {m_ut_def} {m_ut_tun} {m_ut_msk} | {m_leak}")
    print("="*110)

    # Save summary json
    summary_path = os.path.join(args.output_dir, "phase_c_summary.json")
    clean_eval = {
        m: {
            k: {
                "macro_f1": v["macro_f1"],
                "micro_f1": v["micro_f1"],
                "per_class": v["per_class"],
            }
            for k, v in all_eval_results[m].items() if isinstance(v, dict) and "macro_f1" in v
        } for m in all_eval_results
    }
    with open(summary_path, "w") as f:
        json.dump(clean_eval, f, indent=2)
    print(f"\nSaved all metrics to {summary_path}")


if __name__ == "__main__":
    main()
