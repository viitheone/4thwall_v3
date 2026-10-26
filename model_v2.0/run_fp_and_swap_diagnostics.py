"""
Phase C Follow-Up Diagnostics:
1. FP Diagnostics (Row-weighted grouped test, true-Normal rows):
   - Per-label FP counts at threshold 0.5.
   - FPR counting all 12 attack labels vs excluding 272 - Protocol Manipulation.
   - Top 20 normalized templates by FP count among Normal rows.
   - Fraction of Normal-row FPs explained by top 10 and top 100 templates.
   - UA-group and date-bucket FPR tables with 272 excluded.
2. UA-Swap Robustness:
   - For each class, take row-weighted test positives detected correctly at tuned thresholds.
   - Replace [UA] with each of 5 common browser UAs from Normal rows outside Firefox-70.
   - Re-serialize and re-score with Normal model.
   - Report mean retained recall across 5 UAs vs UA-masked model's recall under its own masking.
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
from sklearn.metrics import recall_score
from transformers import AutoTokenizer, AutoModel

# ── Config & Constants ────────────────────────────────────────────────
BASE_MODEL = "distilbert-base-uncased"
MAX_LENGTH = 256
BATCH_SIZE = 128
SIGMOID_THRESHOLD = 0.5
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

# Phase C Tuned Thresholds from Validation Grid Search
TUNED_THRESHOLDS = {
    '000 - Normal': 0.20,
    '272 - Protocol Manipulation': 0.90,
    '242 - Code Injection': 0.70,
    '88 - OS Command Injection': 0.90,
    '126 - Path Traversal': 0.85,
    '66 - SQL Injection': 0.55,
    '16 - Dictionary-based Password Attack': 0.50,
    '310 - Scanning for Vulnerable Software': 0.50,
    '153 - Input Data Manipulation': 0.50,
    '274 - HTTP Verb Tampering': 0.50,
    '194 - Fake the Source of Data': 0.90,
    '34 - HTTP Response Splitting': 0.50,
    '33 - HTTP Request Smuggling': 0.75,
}

SWAP_UAS = [
    ("Firefox 73 (Linux)", "Mozilla/5.0 (X11; Linux x86_64; rv:73.0) Gecko/20100101 Firefox/73.0"),
    ("MSIE 7.0 (Win 7)", "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.1; Trident/6.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; Media Center PC 6.0; .NET4.0C; .NET4.0E)"),
    ("Chromium 37 (Ubuntu)", "Mozilla/5.0 (X11; Linux i686) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/37.0.2062.120 Chrome/37.0.2062.120 Safari/537.36"),
    ("Safari 8 (Mac OS X)", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/600.2.5 (KHTML, like Gecko) Version/8.0 Safari/600.1.25"),
    ("IE 11 (Win 8.1)", "Mozilla/5.0 (Windows NT 6.3; Win64; x64; Trident/7.0; Touch; MAARJS; rv:11.0) like Gecko"),
]


# ── Model Definition ──────────────────────────────────────────────────
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


# ── Serialization & Key Building ──────────────────────────────────────
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


# ── Model Scoring Function ────────────────────────────────────────────
def score_texts(model, tokenizer, texts, device, desc="Scoring"):
    model.eval()
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(model)
    else:
        net = model

    ds = WAFDataset(texts, None, tokenizer, MAX_LENGTH)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

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


# ── Main Diagnostics ──────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Phase C FP & UA-Swap Diagnostics")
    parser.add_argument("--data_path", default="data/data_capec_multilabel.csv",
                        help="Path to data_capec_multilabel.csv")
    parser.add_argument("--splits_dir", default="outputs/splits",
                        help="Path to precomputed split indices")
    parser.add_argument("--normal_weights", default="outputs/phase_c/normal_model_weights.pt",
                        help="Path to trained normal model weights")
    parser.add_argument("--ua_masked_weights", default="outputs/phase_c/ua_masked_model_weights.pt",
                        help="Path to trained UA-masked model weights")
    parser.add_argument("--output_json", default="outputs/phase_c/fp_and_swap_diagnostics.json",
                        help="Output path for JSON diagnostics report")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, Available GPUs: {torch.cuda.device_count()}")

    # 1. Load Data & Clean
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

    # 2. Load Test Split
    test_idx_path = os.path.join(args.splits_dir, "grouped_test_idx.npy")
    if not os.path.exists(test_idx_path):
        # Check zip archive
        zip_path = os.path.join("outputs", "splits_phase_c.zip")
        if os.path.exists(zip_path):
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(args.splits_dir)

    test_idx = np.load(test_idx_path)
    df_test = df.iloc[test_idx].copy().reset_index(drop=True)
    test_labels = labels_np[test_idx]
    print(f"Loaded Row-Weighted Test Set: {len(df_test):,d} samples.")

    # 3. Load Models
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    print(f"\nLoading Normal Model from {args.normal_weights}...")
    normal_model = MultiLabelWAFModel().to(device)
    normal_model.load_state_dict(torch.load(args.normal_weights, map_location=device))
    normal_model.eval()

    print(f"Loading UA-Masked Model from {args.ua_masked_weights}...")
    ua_masked_model = MultiLabelWAFModel().to(device)
    ua_masked_model.load_state_dict(torch.load(args.ua_masked_weights, map_location=device))
    ua_masked_model.eval()

    # 4. Serialize & Score Row-Weighted Test Set with Normal Model
    print("\nSerializing original test strings...")
    test_texts = fast_serialize(df_test, mask_ua=False)
    print("Scoring test set with Normal Model...")
    test_probs = score_texts(normal_model, tokenizer, test_texts, device, desc="Normal Model [Test]")

    # Identify true Normal rows
    # True Normal: label 0 is 1 and all 12 attack labels are 0
    is_true_normal = (test_labels[:, 0] == 1) & (test_labels[:, 1:].sum(axis=1) == 0)
    normal_indices = np.where(is_true_normal)[0]
    n_normal = len(normal_indices)
    normal_probs = test_probs[normal_indices]
    df_normal = df_test.iloc[normal_indices].copy().reset_index(drop=True)
    normal_texts = [test_texts[i] for i in normal_indices]

    print(f"\n{'='*95}")
    print(f"STEP 1: FP DIAGNOSTICS (ROW-WEIGHTED GROUPED TEST, {n_normal:,d} TRUE-NORMAL ROWS)")
    print(f"{'='*95}")

    # ==================================================================
    # 1.a: Per-Label FP Counts & Overall FPR (All 12 vs Excluding 272)
    # ==================================================================
    print("\n1.a: Per-Label FP Counts at Threshold 0.5 among True-Normal Rows:")
    print(f"{'Class':<42s} {'FP Count':>12s} {'FPR on Normal (%)':>20s}")
    print("-" * 76)

    per_label_fps = {}
    for j in range(1, 13):
        col_name = LABEL_COLS[j]
        fp_count = int((normal_probs[:, j] >= SIGMOID_THRESHOLD).sum())
        fpr_pct = 100.0 * fp_count / n_normal
        per_label_fps[col_name] = {"fp_count": fp_count, "fpr_pct": fpr_pct}
        print(f"{col_name:<42s} {fp_count:>12,d} {fpr_pct:>19.2f}%")

    # (i) Counting all 12 attack labels
    any_attack_fp_mask = (normal_probs[:, 1:] >= SIGMOID_THRESHOLD).any(axis=1)
    total_fp_all = int(any_attack_fp_mask.sum())
    fpr_all = 100.0 * total_fp_all / n_normal

    # (ii) Excluding 272 - Protocol Manipulation (class index 1 in full list, index 0 in attack slice)
    attack_indices_no_272 = [j for j in range(1, 13) if j != 1]
    no_272_fp_mask = (normal_probs[:, attack_indices_no_272] >= SIGMOID_THRESHOLD).any(axis=1)
    total_fp_no_272 = int(no_272_fp_mask.sum())
    fpr_no_272 = 100.0 * total_fp_no_272 / n_normal

    print("-" * 76)
    print(f"{'OVERALL (Counting All 12 Attack Labels)':<42s} {total_fp_all:>12,d} {fpr_all:>19.2f}%")
    print(f"{'OVERALL (EXCLUDING 272 - Protocol Manipulation)':<42s} {total_fp_no_272:>12,d} {fpr_no_272:>19.2f}%")
    delta_fp = total_fp_all - total_fp_no_272
    print(f"\n[KEY TAKEAWAY] 272 - Protocol Manipulation alone is responsible for {delta_fp:,d} FPs ({100.0*delta_fp/total_fp_all:.1f}% of all false positives)!")

    # ==================================================================
    # 1.b & 1.c: Top Normalized Templates by FP Count
    # ==================================================================
    print(f"\n{'='*95}")
    print("1.b: TOP 20 NORMALIZED TEMPLATES BY FP COUNT AMONG TRUE-NORMAL ROWS")
    print(f"{'='*95}")

    print("Computing normalized keys for true-normal test rows...")
    normal_keys = [build_normalized_key(t) for t in normal_texts]
    df_normal['norm_key'] = normal_keys
    df_normal['is_fp'] = any_attack_fp_mask
    df_normal['raw_text'] = normal_texts

    # Also determine which attack labels fired for each row
    fired_labels_per_row = []
    for row_probs in normal_probs[:, 1:]:
        fired = [LABEL_COLS[j + 1].split(' - ')[1] for j, p in enumerate(row_probs) if p >= SIGMOID_THRESHOLD]
        fired_labels_per_row.append(', '.join(fired) if fired else 'None')
    df_normal['fired_labels'] = fired_labels_per_row

    # Group by norm_key among true-Normal rows
    grouped_templates = df_normal.groupby('norm_key').agg(
        rows_in_test=('is_fp', 'count'),
        fp_count=('is_fp', 'sum'),
        fired_labels=('fired_labels', lambda s: pd.Series(s[s != 'None']).value_counts().index[0] if (s != 'None').any() else 'None'),
        ua=('request_user_agent', lambda s: s.iloc[0] if len(s) > 0 else ''),
        example_text=('raw_text', lambda s: s.iloc[0] if len(s) > 0 else '')
    ).reset_index()

    # Sort descending by FP count
    grouped_templates = grouped_templates.sort_values(by=['fp_count', 'rows_in_test'], ascending=False).reset_index(drop=True)

    # 1.c: Concentrations
    top_10_fp_sum = int(grouped_templates.iloc[:10]['fp_count'].sum())
    top_100_fp_sum = int(grouped_templates.iloc[:100]['fp_count'].sum())
    frac_top_10 = 100.0 * top_10_fp_sum / total_fp_all if total_fp_all > 0 else 0.0
    frac_top_100 = 100.0 * top_100_fp_sum / total_fp_all if total_fp_all > 0 else 0.0

    print(f"\n1.c: Concentration of False Positives across Templates:")
    print(f"  Total Normal-row FPs: {total_fp_all:,d}")
    print(f"  Top 10 FP Templates:  {top_10_fp_sum:,d} FPs ({frac_top_10:.2f}% of all FPs)")
    print(f"  Top 100 FP Templates: {top_100_fp_sum:,d} FPs ({frac_top_100:.2f}% of all FPs)")
    print("-" * 95)

    print("\nTop 20 FP Templates Table:")
    print(f"{'#':<3s} {'Template Key (truncated)':<42s} {'Rows':>6s} {'FPs':>6s} {'Firing Label':<22s} {'User-Agent (truncated)':<30s}")
    print("-" * 115)

    top_20_records = []
    for rank in range(min(20, len(grouped_templates))):
        row = grouped_templates.iloc[rank]
        key_trunc = row['norm_key'][:40] + ('..' if len(row['norm_key']) > 40 else '')
        ua_trunc = str(row['ua'])[:28] + ('..' if len(str(row['ua'])) > 28 else '')
        ex_150 = str(row['example_text'])[:150]
        label_fired = str(row['fired_labels'])[:20]

        print(f"{rank+1:<3d} {key_trunc:<42s} {row['rows_in_test']:>6,d} {row['fp_count']:>6,d} {label_fired:<22s} {ua_trunc:<30s}")
        print(f"    Example: \"{ex_150}\"")

        top_20_records.append({
            "rank": rank + 1,
            "norm_key": row['norm_key'],
            "rows_in_test": int(row['rows_in_test']),
            "fp_count": int(row['fp_count']),
            "fired_labels": str(row['fired_labels']),
            "ua": str(row['ua']),
            "example_150": ex_150
        })

    # ==================================================================
    # 1.d: Repeat UA-Group & Date-Bucket Tables (With and Without 272)
    # ==================================================================
    print(f"\n{'='*95}")
    print("1.d: UA-GROUP & DATE-BUCKET FPR TABLES (WITH VS WITHOUT 272)")
    print(f"{'='*95}")

    df_normal['is_fp_all'] = any_attack_fp_mask
    df_normal['is_fp_no_272'] = no_272_fp_mask

    # (i) UA-Group
    ff70_ua = 'Mozilla/5.0 (X11; Linux x86_64; rv:70.0) Gecko/20100101 Firefox/70.0'
    is_ff70 = df_normal['request_user_agent'] == ff70_ua

    ff_cnt = int(is_ff70.sum())
    ff_fp_all = int(df_normal.loc[is_ff70, 'is_fp_all'].sum())
    ff_fp_no = int(df_normal.loc[is_ff70, 'is_fp_no_272'].sum())

    other_cnt = int((~is_ff70).sum())
    other_fp_all = int(df_normal.loc[~is_ff70, 'is_fp_all'].sum())
    other_fp_no = int(df_normal.loc[~is_ff70, 'is_fp_no_272'].sum())

    print(f"\n(i) False Positive Rate by User-Agent Group:")
    print(f"{'Group':<32s} {'Normal Rows':>12s} | {'FPs (All)':>10s} {'FPR (All)':>10s} | {'FPs (No 272)':>12s} {'FPR (No 272)':>12s}")
    print("-" * 88)
    print(f"{'Firefox 70 (Campaign UA)':<32s} {ff_cnt:>12,d} | {ff_fp_all:>10,d} {100.0*ff_fp_all/ff_cnt:>9.2f}% | {ff_fp_no:>12,d} {100.0*ff_fp_no/ff_cnt:>11.2f}%")
    print(f"{'All Other User-Agents':<32s} {other_cnt:>12,d} | {other_fp_all:>10,d} {100.0*other_fp_all/other_cnt:>9.2f}% | {other_fp_no:>12,d} {100.0*other_fp_no/other_cnt:>11.2f}%")

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
        b_fp_all = int(sub_b['is_fp_all'].sum())
        b_fp_no = int(sub_b['is_fp_no_272'].sum())
        fpr_a = 100.0 * b_fp_all / b_cnt if b_cnt > 0 else 0.0
        fpr_n = 100.0 * b_fp_no / b_cnt if b_cnt > 0 else 0.0
        print(f"{b_name:<32s} {b_cnt:>12,d} | {b_fp_all:>10,d} {fpr_a:>9.2f}% | {b_fp_no:>12,d} {fpr_n:>11.2f}%")


    # ==================================================================
    # STEP 2: UA-SWAP ROBUSTNESS
    # ==================================================================
    print(f"\n\n{'='*95}")
    print("STEP 2: UA-SWAP ROBUSTNESS TEST")
    print(f"{'='*95}")
    print("5 Common Browser User-Agents Sampled from Normal Rows Outside Firefox 70:")
    for idx, (label_ua, val_ua) in enumerate(SWAP_UAS):
        print(f"  [{idx+1}] {label_ua:<22s}: \"{val_ua}\"")

    print(f"\nEvaluating retained recall when [UA] is swapped with each browser UA...")
    swap_results = {}

    print(f"\n{'Class':<40s} {'Thresh':>7s} {'Orig TP':>8s} | {'Mean Swap Recall':>16s} | {'UA-Masked Recall':>17s} | {'Swap Delta':>10s}")
    print("-" * 105)

    for j, col in enumerate(LABEL_COLS):
        thresh = TUNED_THRESHOLDS[col]
        # Identify row-weighted test positives detected correctly by the Normal model at its tuned threshold
        tp_mask = (test_labels[:, j] == 1) & (test_probs[:, j] >= thresh)
        orig_tp = int(tp_mask.sum())

        if orig_tp == 0:
            print(f"{col:<40s} {thresh:>7.2f} {orig_tp:>8,d} | {'n/a':>16s} | {'n/a':>17s} | {'n/a':>10s}")
            continue

        df_tp = df_test.iloc[tp_mask].copy()

        # Score with 5 swapped UAs using Normal model
        swapped_recalls = []
        for name_ua, val_ua in SWAP_UAS:
            df_swapped = df_tp.copy()
            df_swapped['request_user_agent'] = val_ua
            swapped_texts = fast_serialize(df_swapped, mask_ua=False)
            probs_swapped = score_texts(normal_model, tokenizer, swapped_texts, device, desc=f"Swap {col[:15]}")
            detected = int((probs_swapped[:, j] >= thresh).sum())
            swapped_recalls.append(detected / orig_tp)

        mean_swap_recall = float(np.mean(swapped_recalls))

        # Score with UA-Masked Model under its own masking ([UA] masked)
        df_masked = df_tp.copy()
        masked_texts = fast_serialize(df_masked, mask_ua=True)
        probs_masked = score_texts(ua_masked_model, tokenizer, masked_texts, device, desc=f"Mask {col[:15]}")
        masked_detected = int((probs_masked[:, j] >= thresh).sum())
        masked_recall = float(masked_detected / orig_tp)

        delta_swap = (mean_swap_recall - 1.0) * 100.0

        swap_results[col] = {
            "threshold": thresh,
            "orig_tp": orig_tp,
            "mean_swapped_recall": mean_swap_recall,
            "ua_masked_recall": masked_recall,
            "per_ua_recalls": swapped_recalls
        }

        print(f"{col:<40s} {thresh:>7.2f} {orig_tp:>8,d} | {mean_swap_recall*100:>15.2f}% | {masked_recall*100:>16.2f}% | {delta_swap:>9.2f}%")

    print("-" * 105)

    # Save summary JSON
    out_dir = os.path.dirname(args.output_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    summary_data = {
        "step_1_fp_diagnostics": {
            "total_normal_rows": n_normal,
            "overall_fp_all_12": {"count": total_fp_all, "fpr_pct": fpr_all},
            "overall_fp_excl_272": {"count": total_fp_no_272, "fpr_pct": fpr_no_272},
            "per_label_fps": per_label_fps,
            "concentration": {
                "top_10_fp_pct": frac_top_10,
                "top_100_fp_pct": frac_top_100
            },
            "top_20_templates": top_20_records
        },
        "step_2_ua_swap_robustness": swap_results
    }
    with open(args.output_json, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\nSaved full diagnostics report to {args.output_json}")


if __name__ == "__main__":
    main()
