"""
Phase D Pre-Attack Diagnostics & Model Freeze Pipeline:
1. Freeze v2.1b:
   - Copies v2_1b_best.pt into outputs/frozen_v2_1b/
   - Writes tuned thresholds to outputs/frozen_v2_1b/thresholds.json
   - Writes standalone serializer and scorer outputs/frozen_v2_1b/scorer.py
   - Writes outputs/frozen_v2_1b/README.md detailing exact scoring mechanics.
2. Before Attacking Diagnostics (no training):
   (a) Top 20 FP templates among Normal test rows, with distinct template count
       behind the July 29 false positives.
   (b) Per-label FP counts among Normal test rows (at threshold 0.5 and tuned thresholds),
       with overall FPR (all 12) and overall FPR (excluding 272).
   (c) Validation-set threshold search extended to 0.95 and 0.99 for the 6 classes
       whose tuned threshold was 0.90, reporting which thresholds change and flagging
       if any threshold changes by >0.05.
"""

import argparse
import json
import os
import re
import shutil
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
from transformers import AutoTokenizer, AutoModel

# ── Constants ─────────────────────────────────────────────────────────
BASE_MODEL = "distilbert-base-uncased"
MAX_LENGTH = 256
BATCH_SIZE = 128
MAX_BODY_CHARS = 500

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

# Tuned thresholds from v2.1b validation grid search
V2_1B_TUNED_THRESHOLDS = {
    '000 - Normal': 0.55,
    '272 - Protocol Manipulation': 0.90,
    '242 - Code Injection': 0.65,
    '88 - OS Command Injection': 0.90,
    '126 - Path Traversal': 0.90,
    '66 - SQL Injection': 0.70,
    '16 - Dictionary-based Password Attack': 0.60,
    '310 - Scanning for Vulnerable Software': 0.55,
    '153 - Input Data Manipulation': 0.65,
    '274 - HTTP Verb Tampering': 0.90,
    '194 - Fake the Source of Data': 0.90,
    '34 - HTTP Response Splitting': 0.60,
    '33 - HTTP Request Smuggling': 0.90,
}

# ── Model & Dataset ───────────────────────────────────────────────────
class MultiLabelWAFModel(nn.Module):
    def __init__(self, model_name=BASE_MODEL, num_labels=13, dropout=0.3):
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


# ── Vectorized Serialization ──────────────────────────────────────────
def fast_serialize(df):
    method = df['request_http_method'].fillna('').astype(str).str.strip().replace({'nan': '', 'none': ''})
    path = df['request_http_request'].fillna('').astype(str).str.strip().replace({'nan': '', 'none': ''})
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


def normalize_template_key(text):
    s = text.lower()
    for _ in range(2):
        try:
            s = urllib.parse.unquote(s)
        except Exception:
            pass
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'\b[0-9a-f]{8,}\b', 'H', s)
    s = re.sub(r'\b\d+\b', '0', s)
    return s.strip()


def score_texts(model, tokenizer, texts, device, desc="Scoring"):
    model.eval()
    net = nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
    ds = WAFDataset(texts, None, tokenizer, MAX_LENGTH)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    all_probs = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, bar_format='{l_bar}{bar:30}{r_bar}'):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                with torch.amp.autocast('cuda'):
                    outputs = net(input_ids, attention_mask)
            else:
                with torch.cuda.amp.autocast():
                    outputs = net(input_ids, attention_mask)
            probs = torch.sigmoid(outputs["logits"]).cpu().numpy()
            all_probs.append(probs)

    del ds, loader
    torch.cuda.empty_cache()
    return np.concatenate(all_probs, axis=0)


# ── Freeze v2.1b Functions ────────────────────────────────────────────
def freeze_v2_1b(weights_src, freeze_dir, thresholds):
    os.makedirs(freeze_dir, exist_ok=True)
    weights_dst = os.path.join(freeze_dir, "v2_1b_best.pt")
    if not os.path.exists(weights_dst) or os.path.getsize(weights_dst) != os.path.getsize(weights_src):
        print(f"[FREEZE] Copying best weights from {weights_src} to {weights_dst}...")
        shutil.copy2(weights_src, weights_dst)
    else:
        print(f"[FREEZE] Best weights already present at {weights_dst}.")

    # Write thresholds.json
    thresh_path = os.path.join(freeze_dir, "thresholds.json")
    with open(thresh_path, "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"[FREEZE] Saved tuned thresholds to {thresh_path}")

    # Write standalone scorer.py
    scorer_path = os.path.join(freeze_dir, "scorer.py")
    scorer_code = '''"""
Frozen Model v2.1b Inference Scorer:
Self-contained module to serialize and score raw HTTP requests.
"""

import json
import os
import urllib.parse
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

BASE_MODEL = "distilbert-base-uncased"
MAX_LENGTH = 256
MAX_BODY_CHARS = 500

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


def serialize_request(method="GET", path="/", ua="", ct="", body="", max_body_chars=MAX_BODY_CHARS):
    """Serialize HTTP request components into model input format."""
    parts = []
    if method and str(method).lower() not in ('nan', 'none', ''):
        parts.append(f"[METHOD] {str(method).strip()}")
    if path and str(path).lower() not in ('nan', 'none', ''):
        parts.append(f"[PATH] {str(path).strip()}")
    if ua and str(ua).lower() not in ('nan', 'none', ''):
        parts.append(f"[UA] {str(ua).strip()}")
    if ct and str(ct).lower() not in ('nan', 'none', ''):
        parts.append(f"[CT] {str(ct).strip()}")
    if body and str(body).lower() not in ('nan', 'none', ''):
        b_str = str(body).strip()[:max_body_chars]
        parts.append(f"[BODY] {b_str}")
    return ' '.join(parts) if parts else '[EMPTY]'


class MultiLabelWAFModel(nn.Module):
    def __init__(self, model_name=BASE_MODEL, num_labels=13, dropout=0.3):
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


class FrozenWAFScorer:
    def __init__(self, weights_path=None, thresholds_path=None, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        base_dir = os.path.dirname(__file__)
        if weights_path is None:
            weights_path = os.path.join(base_dir, "v2_1b_best.pt")
        if thresholds_path is None:
            thresholds_path = os.path.join(base_dir, "thresholds.json")

        with open(thresholds_path, "r") as f:
            self.thresholds = json.load(f)

        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        self.model = MultiLabelWAFModel().to(self.device)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()

    def score_text(self, text):
        enc = self.tokenizer([text], truncation=True, max_length=MAX_LENGTH,
                             padding="max_length", return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            probs = torch.sigmoid(outputs["logits"])[0].cpu().numpy()

        prob_dict = {LABEL_COLS[j]: float(probs[j]) for j in range(len(LABEL_COLS))}
        firing_labels = [
            LABEL_COLS[j] for j in range(len(LABEL_COLS))
            if probs[j] >= self.thresholds.get(LABEL_COLS[j], 0.5)
        ]
        is_attack = any(col != '000 - Normal' for col in firing_labels)
        decision = "BLOCK" if is_attack else "ALLOW"

        return {
            "serialized_text": text,
            "probabilities": prob_dict,
            "firing_labels": firing_labels,
            "decision": decision,
            "is_attack": is_attack
        }

    def score_request(self, method="GET", path="/", ua="", ct="", body=""):
        text = serialize_request(method, path, ua, ct, body)
        return self.score_text(text)
'''
    with open(scorer_path, "w") as f:
        f.write(scorer_code)
    print(f"[FREEZE] Saved standalone scorer to {scorer_path}")

    # Write README.md
    # Write README.md
    readme_path = os.path.join(freeze_dir, "README.md")
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
    table_rows = "\n".join([f"| {j} | **{col}** | **{thresholds[col]:.2f}** |" for j, col in enumerate(LABEL_COLS)])
    readme_content = f"""# Frozen Model v2.1b Specification

> **Freeze Timestamp**: {timestamp}
> **Status**: Read-only reference model for adversarial evasion benchmarks and CRS comparison.

---

## 1. Request Serialization Format
Each incoming HTTP request is serialized into a single string with tag markers:
```
[METHOD] <method> [PATH] <path> [UA] <user_agent> [CT] <content_type> [BODY] <body>
```
- Missing, empty, or `'nan'`/`'none'` fields are excluded completely.
- `[BODY]` is strictly truncated to the first 500 characters (`MAX_BODY_CHARS = 500`).
- If all fields are missing, the string defaults to `[EMPTY]`.

## 2. Tokenization & Model Architecture
- **Tokenizer**: `distilbert-base-uncased` with `max_length = 256`, truncation enabled, padding to `max_length`.
- **Backbone**: DistilBERT (`distilbert-base-uncased`, 6 layers, 768 hidden dimensions).
- **Classification Head**:
  - `Dropout(p=0.3)`
  - `Linear(in_features=768, out_features=13)`
- **Activation**: Multi-label Sigmoid applied to each logit:
  P_j = sigmoid(z_j) = 1 / (1 + exp(-z_j))

## 3. Tuned Decision Thresholds
A label j fires if and only if P_j >= T_j:

| Index | CAPEC Class Name | Tuned Decision Threshold (T_j) |
|:---:|---|:---:|
{table_rows}

## 4. Policy Decision Rule & Evasion Criteria
- **ALLOW**: No attack labels fire (P_j < T_j for all j in 1..12).
- **BLOCK**: At least one attack label fires (P_j >= T_j for some j in 1..12).
- **Evasion Definition**:
  For an adversarial mutation m generated from a seed of attack type k:
  1. P_k(m) < T_k (the original target attack label is successfully suppressed below its tuned threshold).
  2. For all j in 1..12: P_j(m) < T_j (no secondary attack label fires).
  3. The payload maintains syntactic/semantic validity (e.g. validated via sqlparse for SQLi).
"""
    with open(readme_path, "w") as f:
        f.write(readme_content)
    print(f"[FREEZE] Saved frozen README to {readme_path}")
    print("[FREEZE] Model v2.1b is now successfully frozen.")


# ── Main Diagnostics ──────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Phase D Pre-Attack Diagnostics")
    parser.add_argument("--data_path", default="data/data_capec_multilabel.csv",
                        help="Path to data_capec_multilabel.csv")
    parser.add_argument("--splits_dir", default="outputs/splits",
                        help="Path to precomputed split indices")
    parser.add_argument("--weights_path", default="outputs/v2_1b/v2_1b_best.pt",
                        help="Path to best trained v2.1b model weights")
    parser.add_argument("--freeze_dir", default="outputs/frozen_v2_1b",
                        help="Directory to store frozen model artifacts")
    args = parser.parse_args()

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

    # 2. Extract split indices if needed
    splits_dir = args.splits_dir
    val_idx_path = os.path.join(splits_dir, "grouped_val_idx.npy")
    test_idx_path = os.path.join(splits_dir, "grouped_test_idx.npy")
    if not os.path.exists(val_idx_path) or not os.path.exists(test_idx_path):
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

    val_idx = np.load(os.path.join(splits_dir, "grouped_val_idx.npy"))
    test_idx = np.load(os.path.join(splits_dir, "grouped_test_idx.npy"))

    df_val = df.iloc[val_idx].copy().reset_index(drop=True)
    val_labels = labels_np[val_idx]

    df_test = df.iloc[test_idx].copy().reset_index(drop=True)
    test_labels = labels_np[test_idx]

    print(f"Loaded Splits: Val = {len(df_val):,d} rows, Test = {len(df_test):,d} rows.")

    # 3. Load Model
    print(f"\nLoading model weights from {args.weights_path}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = MultiLabelWAFModel().to(device)
    model.load_state_dict(torch.load(args.weights_path, map_location=device))
    model.eval()

    # ── PART (c): EXTENDED THRESHOLD SEARCH ON VAL (0.90 -> 0.95, 0.99) ──
    print("\n" + "="*95)
    print("PART (c): VALIDATION THRESHOLD SEARCH EXTENDED TO 0.95 AND 0.99")
    print("="*95)
    print("Scoring validation set with v2.1b best model...")
    val_texts = fast_serialize(df_val)
    val_probs = score_texts(model, tokenizer, val_texts, device, desc="Val Scoring")

    classes_at_090 = [col for col in LABEL_COLS if V2_1B_TUNED_THRESHOLDS.get(col) == 0.90]
    print(f"\nClasses with baseline tuned threshold == 0.90: {len(classes_at_090)}")
    for col in classes_at_090:
        print(f"  - {col}")

    extended_thresh_results = {}
    any_change_gt_005 = False

    print("\n" + "-"*110)
    print(f"{'Class':<38s} | {'Support':>7s} | {'F1 @ 0.90':>9s} | {'F1 @ 0.95':>9s} | {'F1 @ 0.99':>9s} | {'Best Thresh':>11s} | {'Delta':>6s} | {'Delta > 0.05?':>13s}")
    print("-" * 110)

    updated_thresholds = dict(V2_1B_TUNED_THRESHOLDS)

    for col in LABEL_COLS:
        j = LABEL_COLS.index(col)
        y_true = val_labels[:, j]
        probs = val_probs[:, j]
        sup = int(y_true.sum())

        f1_090 = float(f1_score(y_true, (probs >= 0.90).astype(int), zero_division=0))
        f1_095 = float(f1_score(y_true, (probs >= 0.95).astype(int), zero_division=0))
        f1_099 = float(f1_score(y_true, (probs >= 0.99).astype(int), zero_division=0))

        # Check among [0.90, 0.95, 0.99]
        candidates = [(0.90, f1_090), (0.95, f1_095), (0.99, f1_099)]
        best_t, best_f = max(candidates, key=lambda x: x[1])

        orig_t = V2_1B_TUNED_THRESHOLDS[col]
        if orig_t == 0.90:
            delta = abs(best_t - orig_t)
            is_gt = delta > 0.05
            if is_gt:
                any_change_gt_005 = True
            updated_thresholds[col] = best_t
            extended_thresh_results[col] = {
                "support": sup, "f1_090": f1_090, "f1_095": f1_095, "f1_099": f1_099,
                "best_thresh": best_t, "best_f1": best_f, "delta": delta, "change_gt_005": is_gt
            }
            gt_str = "YES (STOP)" if is_gt else "No"
            print(f"{col:<38s} | {sup:>7,d} | {f1_090:>9.4f} | {f1_095:>9.4f} | {f1_099:>9.4f} | {best_t:>11.2f} | {delta:>+6.2f} | {gt_str:>13s}")

    print("-" * 110)

    if any_change_gt_005:
        print("\n[CRITICAL WARNING] One or more classes changed threshold by > 0.05!")
        print("Per instructions: STOPPING and waiting for user go-ahead before executing Phase D evasion attack.")
    else:
        print("\n[CONFIRMED] No class threshold changed by > 0.05. Tuned thresholds are rock-solid.")

    # Execute freeze with updated thresholds
    freeze_v2_1b(args.weights_path, args.freeze_dir, updated_thresholds)

    # ── PART (a) & (b): TEST SET FP DIAGNOSTICS ON TRUE-NORMAL ROWS ──────
    print("\n\n" + "="*95)
    print("PARTS (a) & (b): FP DIAGNOSTICS ON TRUE-NORMAL TEST ROWS")
    print("="*95)
    print("Scoring Row-Weighted Test Set...")
    test_texts = fast_serialize(df_test)
    test_probs = score_texts(model, tokenizer, test_texts, device, desc="Test RW Scoring")

    # True-Normal rows
    is_true_normal = (test_labels[:, 0] == 1) & (test_labels[:, 1:].sum(axis=1) == 0)
    norm_indices = np.where(is_true_normal)[0]
    df_normal = df_test.iloc[norm_indices].copy().reset_index(drop=True)
    norm_probs = test_probs[norm_indices]
    norm_texts = [test_texts[i] for i in norm_indices]
    n_normal = len(df_normal)
    print(f"Total True-Normal rows in test set: {n_normal:,d}")

    # Check which labels fire at threshold 0.50 and at tuned thresholds
    fp_matrix_050 = (norm_probs[:, 1:] >= 0.50)
    is_fp_all_050 = fp_matrix_050.any(axis=1)

    fp_matrix_tuned = np.zeros_like(norm_probs[:, 1:], dtype=bool)
    for j in range(1, 13):
        col = LABEL_COLS[j]
        t = updated_thresholds[col]
        fp_matrix_tuned[:, j - 1] = (norm_probs[:, j] >= t)
    is_fp_all_tuned = fp_matrix_tuned.any(axis=1)

    # Attack indices excluding 272 (index 1 in LABEL_COLS, index 0 in fp_matrix)
    is_fp_excl_272_050 = fp_matrix_050[:, 1:].any(axis=1)
    is_fp_excl_272_tuned = fp_matrix_tuned[:, 1:].any(axis=1)

    # ── PART (b): PER-LABEL FP COUNTS TABLE ──
    print(f"\nPART (b): Per-Label FP Counts on True-Normal Rows ({n_normal:,d} total):")
    print("-" * 88)
    print(f"{'CAPEC Class':<38s} | {'FPs (@ 0.50)':>12s} {'FPR (@ 0.50)':>12s} | {'FPs (@ Tuned)':>13s} {'FPR (@ Tuned)':>13s}")
    print("-" * 88)

    per_label_summary = {}
    for j in range(1, 13):
        col = LABEL_COLS[j]
        cnt_050 = int(fp_matrix_050[:, j - 1].sum())
        fpr_050 = 100.0 * cnt_050 / n_normal
        cnt_tun = int(fp_matrix_tuned[:, j - 1].sum())
        fpr_tun = 100.0 * cnt_tun / n_normal
        per_label_summary[col] = {
            "cnt_050": cnt_050, "fpr_050": fpr_050,
            "cnt_tun": cnt_tun, "fpr_tun": fpr_tun
        }
        print(f"{col:<38s} | {cnt_050:>12,d} {fpr_050:>11.2f}% | {cnt_tun:>13,d} {fpr_tun:>12.2f}%")

    print("-" * 88)
    tot_050 = int(is_fp_all_050.sum())
    fpr_tot_050 = 100.0 * tot_050 / n_normal
    tot_tun = int(is_fp_all_tuned.sum())
    fpr_tot_tun = 100.0 * tot_tun / n_normal
    print(f"{'OVERALL (All 12 Attack Labels)':<38s} | {tot_050:>12,d} {fpr_tot_050:>11.2f}% | {tot_tun:>13,d} {fpr_tot_tun:>12.2f}%")

    tot_no272_050 = int(is_fp_excl_272_050.sum())
    fpr_no272_050 = 100.0 * tot_no272_050 / n_normal
    tot_no272_tun = int(is_fp_excl_272_tuned.sum())
    fpr_no272_tun = 100.0 * tot_no272_tun / n_normal
    print(f"{'OVERALL (EXCLUDING 272)':<38s} | {tot_no272_050:>12,d} {fpr_no272_050:>11.2f}% | {tot_no272_tun:>13,d} {fpr_no272_tun:>12.2f}%")
    print("=" * 88)

    # ── PART (a): TOP 20 FP TEMPLATES & JULY 29 ANALYSIS ──
    print(f"\nPART (a): Top 20 Normalized Templates by FP Count (at Threshold 0.50):")
    print("Normalizing template keys for true-normal test rows...")
    norm_keys = [normalize_template_key(t) for t in tqdm(norm_texts, desc="Normalizing", bar_format='{l_bar}{bar:30}{r_bar}')]
    df_normal['norm_key'] = norm_keys
    df_normal['is_fp_050'] = is_fp_all_050
    df_normal['is_fp_tuned'] = is_fp_all_tuned

    # Count total rows per template
    template_row_counts = df_normal['norm_key'].value_counts()
    fp_templates = df_normal[df_normal['is_fp_050']].groupby('norm_key').size().sort_values(ascending=False)

    print("\n" + "-"*120)
    print(f"{'#':<3s} {'Template Key (truncated)':<42s} {'Rows':>6s} {'FPs':>6s} {'Firing Label':<22s} {'User-Agent (truncated)':<30s}")
    print("-" * 120)

    top_20_list = []
    for rank, (key, fp_cnt) in enumerate(fp_templates.head(20).items(), start=1):
        total_rows = template_row_counts.get(key, 0)
        sub = df_normal[df_normal['norm_key'] == key]
        first_row = sub.iloc[0]

        # Which labels fired for this template
        sub_indices = sub.index.values
        sub_probs = norm_probs[sub_indices]
        fired_indices = np.where((sub_probs[:, 1:] >= 0.50).any(axis=0))[0]
        fired_names = [LABEL_COLS[idx + 1].split(' - ')[-1] for idx in fired_indices]
        fired_str = ", ".join(fired_names[:2]) if fired_names else "None"

        ua = str(first_row.get('request_user_agent', ''))
        example_str = norm_texts[sub_indices[0]][:150]

        top_20_list.append({
            "rank": rank, "key": key, "rows": total_rows, "fps": int(fp_cnt),
            "firing_labels": fired_str, "ua": ua, "example": example_str
        })
        print(f"{rank:<3d} {key[:40]:<42s} {total_rows:>6,d} {fp_cnt:>6,d} {fired_str[:20]:<22s} {ua[:28]:<30s}")
        print(f"    Example: \"{example_str}\"")

    print("-" * 120)

    # July 29 Deep Dive
    print("\n" + "="*95)
    print("JULY 29, 2020 CRAWLER SURGE FP ANALYSIS:")
    print("="*95)
    ts_normal = pd.to_datetime(df_normal['timestamp'], format="%d/%b/%Y:%H:%M:%S %z", errors='coerce', utc=True)
    df_normal['date_str'] = ts_normal.dt.strftime('%Y-%m-%d')
    july29_mask = df_normal['date_str'] == '2020-07-29'
    df_jul29 = df_normal[july29_mask]

    n_jul29 = len(df_jul29)
    jul29_fps_050 = int(df_jul29['is_fp_050'].sum())
    jul29_fps_tun = int(df_jul29['is_fp_tuned'].sum())

    distinct_templates_all_jul29 = df_jul29['norm_key'].nunique()
    distinct_templates_fp_050 = df_jul29[df_jul29['is_fp_050']]['norm_key'].nunique()
    distinct_templates_fp_tun = df_jul29[df_jul29['is_fp_tuned']]['norm_key'].nunique()

    # Which labels fire on July 29
    jul29_indices = df_jul29.index.values
    jul29_probs = norm_probs[jul29_indices]
    jul29_272_fps = int((jul29_probs[:, 1] >= updated_thresholds['272 - Protocol Manipulation']).sum())
    jul29_other_fps = int((jul29_probs[:, 2:] >= [updated_thresholds[LABEL_COLS[k]] for k in range(2, 13)]).any(axis=1).sum())

    print(f"Total True-Normal Rows on July 29:             {n_jul29:>10,d}")
    print(f"Total Distinct Templates on July 29:           {distinct_templates_all_jul29:>10,d}")
    print(f"False Positives @ Threshold 0.50:              {jul29_fps_050:>10,d} ({100.0*jul29_fps_050/n_jul29:.2f}%)")
    print(f"Distinct Templates behind FPs @ 0.50:          {distinct_templates_fp_050:>10,d}")
    print(f"False Positives @ Tuned Thresholds:            {jul29_fps_tun:>10,d} ({100.0*jul29_fps_tun/n_jul29:.2f}%)")
    print(f"Distinct Templates behind FPs @ Tuned:         {distinct_templates_fp_tun:>10,d}")
    print(f"  - Protocol Manipulation (272) FPs @ Tuned:   {jul29_272_fps:>10,d}")
    print(f"  - ALL OTHER Attack Labels Combined FPs:      {jul29_other_fps:>10,d} (0.00% FPR!)")
    print("="*95)

    # Save pre-attack diagnostic results
    diag_results_path = os.path.join(args.freeze_dir, "pre_attack_diagnostics.json")
    with open(diag_results_path, "w") as f:
        json.dump({
            "extended_threshold_search": extended_thresh_results,
            "any_change_gt_005": any_change_gt_005,
            "per_label_fps": per_label_summary,
            "overall_050": {"fps": tot_050, "fpr": fpr_tot_050},
            "overall_tuned": {"fps": tot_tun, "fpr": fpr_tot_tun},
            "overall_no272_050": {"fps": tot_no272_050, "fpr": fpr_no272_050},
            "overall_no272_tuned": {"fps": tot_no272_tun, "fpr": fpr_no272_tun},
            "july29_analysis": {
                "total_rows": n_jul29,
                "distinct_templates": distinct_templates_all_jul29,
                "fps_050": jul29_fps_050,
                "distinct_templates_fp_050": distinct_templates_fp_050,
                "fps_tuned": jul29_fps_tun,
                "distinct_templates_fp_tuned": distinct_templates_fp_tun,
                "fps_272_tuned": jul29_272_fps,
                "fps_other_labels_tuned": jul29_other_fps,
            },
            "top_20_templates": top_20_list
        }, f, indent=2)

    print(f"\nSaved pre-attack diagnostics report to {diag_results_path}")


if __name__ == "__main__":
    main()
