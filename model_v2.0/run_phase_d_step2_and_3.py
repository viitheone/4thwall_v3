"""
Phase D Execution & Diagnostic Runner:
1. Verifies/updates frozen thresholds (272 -> 0.99, 33 -> 0.95).
2. Reports Class 272 Precision, Recall, F1 at T=0.99 on:
   - Test RW (Row-Weighted grouped test)
   - Test UT (Unique-Template grouped test)
   - Firefox-70 campaign rows in Test RW
   - UA-swap retained recall for 272 at T=0.99 across 5 browser UAs.
3. Samples 50 true-Normal test rows that fire SQLi at T=0.70 and prints:
   - UA, date, 200-char serialized text, and CRS-relevant visible markers.
4. Executes Phase D evasion attack (25 seeds x 150 generations) for SQLi and Code Injection.
"""

import argparse
import json
import os
import re
import sys
import time
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

SWAP_UAS = [
    ("Firefox 73 (Linux)", "Mozilla/5.0 (X11; Linux x86_64; rv:73.0) Gecko/20100101 Firefox/73.0"),
    ("MSIE 7.0 (Win 7)", "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.1; Trident/6.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; Media Center PC 6.0; .NET4.0C; .NET4.0E)"),
    ("Chromium 37 (Ubuntu)", "Mozilla/5.0 (X11; Linux i686) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/37.0.2062.120 Chrome/37.0.2062.120 Safari/537.36"),
    ("Safari 8 (Mac OS X)", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/600.2.5 (KHTML, like Gecko) Version/8.0 Safari/600.1.25"),
    ("IE 11 (Win 8.1)", "Mozilla/5.0 (Windows NT 6.3; Win64; x64; Trident/7.0; Touch; MAARJS; rv:11.0) like Gecko"),
]


# ── Model Architecture ────────────────────────────────────────────────
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
    def __init__(self, texts, tokenizer, max_length=MAX_LENGTH):
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

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }


def fast_serialize(df, override_ua=None):
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


def score_texts(model, tokenizer, texts, device, desc="Scoring"):
    model.eval()
    net = nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
    ds = WAFDataset(texts, tokenizer, MAX_LENGTH)
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


# ── Marker Extraction Helper for SQLi Review ──────────────────────────
def extract_crs_markers(text):
    markers = []
    t_lower = text.lower()

    # Quotes
    quotes = []
    if "'" in text: quotes.append("single_quote(')")
    if '"' in text: quotes.append('double_quote(")')
    if '`' in text: quotes.append('backtick(`)')
    if quotes:
        markers.append("Quotes: " + ", ".join(quotes))

    # SQL Keywords
    keywords = ['union', 'select', 'from', 'where', 'or', 'and', 'order by', 'group by',
                'having', 'limit', 'insert', 'update', 'delete', 'drop', 'sleep', 'benchmark',
                'schema', 'table', 'concat', 'char', 'database', 'version', 'user']
    found_kw = []
    for kw in keywords:
        if re.search(r'\b' + re.escape(kw) + r'\b', t_lower):
            found_kw.append(kw)
    if found_kw:
        markers.append("Keywords: " + ", ".join(found_kw))

    # Tautology / Boolean operators
    tautologies = ['1=1', '0=0', '1>0', '2>1', "'1'='1'", "'a'='a'"]
    found_taut = [t for t in tautologies if t in text or t in text.replace(' ', '')]
    if found_taut:
        markers.append("Tautologies: " + ", ".join(found_taut))

    # Comment tokens
    comments = []
    if '--' in text: comments.append('--')
    if '/*' in text or '*/' in text: comments.append('/*...*/')
    if '#' in text: comments.append('#')
    if comments:
        markers.append("Comments: " + ", ".join(comments))

    # Special characters
    special = []
    if ';' in text: special.append('semicolon(;)')
    if '||' in text: special.append('concat(||)')
    if '&&' in text: special.append('and(&&)')
    if special:
        markers.append("Special: " + ", ".join(special))

    return " | ".join(markers) if markers else "None detected"


# ── Main Runner ───────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Phase D Execution & Diagnostic Runner")
    parser.add_argument("--data_path", default="data/data_capec_multilabel.csv",
                        help="Path to data_capec_multilabel.csv")
    parser.add_argument("--splits_dir", default="outputs/splits",
                        help="Path to precomputed split indices")
    parser.add_argument("--freeze_dir", default="outputs/frozen_v2_1b",
                        help="Path to frozen model directory")
    parser.add_argument("--skip_evasion", action="store_true",
                        help="Skip Phase D mutation evasion attack search")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, Available GPUs: {torch.cuda.device_count()}")

    # ── ITEM 1: VERIFY & PRINT FINAL FROZEN THRESHOLDS ────────────────────
    print("\n" + "="*85)
    print("ITEM 1: FINAL FROZEN MODEL v2.1b DECISION THRESHOLDS")
    print("="*85)
    thresh_path = os.path.join(args.freeze_dir, "thresholds.json")
    if os.path.exists(thresh_path):
        with open(thresh_path, "r") as f:
            thresholds = json.load(f)
    else:
        thresholds = {
            '000 - Normal': 0.55,
            '272 - Protocol Manipulation': 0.99,
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
            '33 - HTTP Request Smuggling': 0.95
        }
        os.makedirs(args.freeze_dir, exist_ok=True)
        with open(thresh_path, "w") as f:
            json.dump(thresholds, f, indent=2)

    # Ensure 272 is 0.99 and 33 is 0.95
    thresholds['272 - Protocol Manipulation'] = 0.99
    thresholds['33 - HTTP Request Smuggling'] = 0.95
    with open(thresh_path, "w") as f:
        json.dump(thresholds, f, indent=2)

    print(f"{'Index':<6s} {'CAPEC Class Name':<42s} {'Frozen Threshold (T_j)':>22s} {'Status':>12s}")
    print("-" * 85)
    for j, col in enumerate(LABEL_COLS):
        status = "UPDATED" if col in ('272 - Protocol Manipulation', '33 - HTTP Request Smuggling') else "Frozen"
        print(f"{j:<6d} {col:<42s} {thresholds[col]:>22.2f} {status:>12s}")
    print("="*85)

    # Load Model
    weights_path = os.path.join(args.freeze_dir, "v2_1b_best.pt")
    if not os.path.exists(weights_path):
        weights_path = "outputs/v2_1b/v2_1b_best.pt"
    print(f"\nLoading frozen model from {weights_path}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = MultiLabelWAFModel().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # Load Dataset
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

    # Load Splits
    test_rw_idx = np.load(os.path.join(args.splits_dir, "grouped_test_idx.npy"))
    test_ut_idx = np.load(os.path.join(args.splits_dir, "grouped_test_unique_template_idx.npy"))

    df_test_rw = df.iloc[test_rw_idx].copy().reset_index(drop=True)
    test_rw_labels = labels_np[test_rw_idx]

    df_test_ut = df.iloc[test_ut_idx].copy().reset_index(drop=True)
    test_ut_labels = labels_np[test_ut_idx]

    # ── ITEM 2: CLASS 272 AT T=0.99 REPORT & UA-SWAP RECALL ───────────────
    print("\n\n" + "="*95)
    print("ITEM 2: CLASS 272 (PROTOCOL MANIPULATION) PERFORMANCE AT T=0.99")
    print("="*95)

    print("Scoring Row-Weighted Test Set...")
    texts_rw = fast_serialize(df_test_rw)
    probs_rw = score_texts(model, tokenizer, texts_rw, device, desc="Test RW Scoring")

    print("Scoring Unique-Template Test Set...")
    texts_ut = fast_serialize(df_test_ut)
    probs_ut = score_texts(model, tokenizer, texts_ut, device, desc="Test UT Scoring")

    idx_272 = 1
    t_272 = 0.99

    # (i) Test RW
    y_rw = test_rw_labels[:, idx_272]
    pred_rw = (probs_rw[:, idx_272] >= t_272).astype(int)
    p_rw = precision_score(y_rw, pred_rw, zero_division=0)
    r_rw = recall_score(y_rw, pred_rw, zero_division=0)
    f_rw = f1_score(y_rw, pred_rw, zero_division=0)
    sup_rw = int(y_rw.sum())
    tp_rw = int((pred_rw & y_rw).sum())
    fp_rw = int((pred_rw & (~y_rw.astype(bool))).sum())

    # (ii) Test UT
    y_ut = test_ut_labels[:, idx_272]
    pred_ut = (probs_ut[:, idx_272] >= t_272).astype(int)
    p_ut = precision_score(y_ut, pred_ut, zero_division=0)
    r_ut = recall_score(y_ut, pred_ut, zero_division=0)
    f_ut = f1_score(y_ut, pred_ut, zero_division=0)
    sup_ut = int(y_ut.sum())
    tp_ut = int((pred_ut & y_ut).sum())
    fp_ut = int((pred_ut & (~y_ut.astype(bool))).sum())

    # (iii) Firefox-70 Campaign Rows in Test RW
    ff70_ua = 'Mozilla/5.0 (X11; Linux x86_64; rv:70.0) Gecko/20100101 Firefox/70.0'
    is_ff70 = df_test_rw['request_user_agent'] == ff70_ua
    y_ff = y_rw[is_ff70]
    pred_ff = pred_rw[is_ff70]
    p_ff = precision_score(y_ff, pred_ff, zero_division=0)
    r_ff = recall_score(y_ff, pred_ff, zero_division=0)
    f_ff = f1_score(y_ff, pred_ff, zero_division=0)
    sup_ff = int(y_ff.sum())
    tp_ff = int((pred_ff & y_ff).sum())
    fp_ff = int((pred_ff & (~y_ff.astype(bool))).sum())

    print(f"\nClass 272 Metrics at Tuned Threshold T = {t_272:.2f}:")
    print("-" * 95)
    print(f"{'Evaluation Split / Slice':<35s} | {'Support':>8s} | {'TP':>6s} | {'FP':>6s} | {'Precision':>10s} | {'Recall':>10s} | {'F1 Score':>10s}")
    print("-" * 95)
    print(f"{'Test RW (Row-Weighted Grouped)':<35s} | {sup_rw:>8,d} | {tp_rw:>6,d} | {fp_rw:>6,d} | {p_rw:>9.4f} | {r_rw:>9.4f} | {f_rw:>9.4f}")
    print(f"{'Test UT (Unique-Template Grouped)':<35s} | {sup_ut:>8,d} | {tp_ut:>6,d} | {fp_ut:>6,d} | {p_ut:>9.4f} | {r_ut:>9.4f} | {f_ut:>9.4f}")
    print(f"{'Firefox-70 Campaign Slice (Test RW)':<35s} | {sup_ff:>8,d} | {tp_ff:>6,d} | {fp_ff:>6,d} | {p_ff:>9.4f} | {r_ff:>9.4f} | {f_ff:>9.4f}")
    print("-" * 95)

    # (iv) UA-Swap Recall for 272 at T=0.99
    tp_mask_99 = (y_rw == 1) & (probs_rw[:, idx_272] >= t_272)
    orig_tp_cnt = int(tp_mask_99.sum())
    print(f"\nEvaluating UA-Swap Recall for Class 272 at T = {t_272:.2f} ({orig_tp_cnt:,d} original TPs)...")

    df_tp_272 = df_test_rw.iloc[tp_mask_99].copy()
    swap_recalls_272 = []
    print("-" * 80)
    print(f"{'Substituted User-Agent':<35s} | {'Retained Detections':>20s} | {'Recall (%)':>12s}")
    print("-" * 80)

    for ua_name, ua_str in SWAP_UAS:
        df_sub = df_tp_272.copy()
        df_sub['request_user_agent'] = ua_str
        sub_texts = fast_serialize(df_sub)
        sub_probs = score_texts(model, tokenizer, sub_texts, device, desc=f"Swap {ua_name[:12]}")
        detected = int((sub_probs[:, idx_272] >= t_272).sum())
        rec = detected / orig_tp_cnt if orig_tp_cnt > 0 else 0.0
        swap_recalls_272.append(rec)
        print(f"{ua_name:<35s} | {detected:>13,d}/{orig_tp_cnt:<6,d} | {rec*100:>11.2f}%")

    mean_swap_272 = float(np.mean(swap_recalls_272))
    print("-" * 80)
    print(f"{'MEAN UA-SWAP RECALL (Class 272 @ 0.99)':<35s} | {' ':>20s} | {mean_swap_272*100:>11.2f}%")
    print("=" * 95)

    # ── ITEM 3: REVIEW OF 50 RANDOM SQLi FALSE POSITIVES AT T=0.70 ────────
    print("\n\n" + "="*110)
    print("ITEM 3: REVIEW OF 50 RANDOM TRUE-NORMAL TEST ROWS FIRING SQLi AT T=0.70")
    print("="*110)

    idx_sqli = 5
    t_sqli = 0.70

    is_true_normal = (test_rw_labels[:, 0] == 1) & (test_rw_labels[:, 1:].sum(axis=1) == 0)
    sqli_fp_mask = is_true_normal & (probs_rw[:, idx_sqli] >= t_sqli)
    sqli_fp_indices = np.where(sqli_fp_mask)[0]
    total_sqli_fps = len(sqli_fp_indices)
    print(f"Total True-Normal rows firing SQL Injection at T={t_sqli:.2f}: {total_sqli_fps:,d}")

    np.random.seed(42)
    sample_size = min(50, total_sqli_fps)
    sampled_indices = np.random.choice(sqli_fp_indices, size=sample_size, replace=False)

    df_sample = df_test_rw.iloc[sampled_indices].copy().reset_index(drop=True)
    sample_texts = [texts_rw[i] for i in sampled_indices]
    sample_probs = probs_rw[sampled_indices, idx_sqli]

    ts_sample = pd.to_datetime(df_sample['timestamp'], format="%d/%b/%Y:%H:%M:%S %z", errors='coerce', utc=True)
    dates_sample = ts_sample.dt.strftime('%Y-%m-%d')

    for i in range(sample_size):
        ua_val = str(df_sample.iloc[i].get('request_user_agent', ''))
        d_val = str(dates_sample.iloc[i])
        raw_text = sample_texts[i]
        text_200 = raw_text[:200]
        score_val = sample_probs[i]
        markers = extract_crs_markers(raw_text)

        print(f"\n--- SQLi FP #{i+1:02d} (Score: {score_val:.4f}) ---")
        print(f"  Date:       {d_val}")
        print(f"  User-Agent: {ua_val}")
        print(f"  Text (200): \"{text_200}\"")
        print(f"  CRS-Relevant Markers: {markers}")

    print("\n" + "="*110)

    # ── ITEM 5: EXECUTE PHASE D EVASION SEARCH ────────────────────────────
    if not args.skip_evasion:
        print("\n\n" + "="*95)
        print("LAUNCHING PHASE D ADVERSARIAL EVASION TEST (phase_d_evasion.py)")
        print("="*95)
        import subprocess
        script_dir = os.path.dirname(os.path.abspath(__file__))
        evasion_script = os.path.join(script_dir, "phase_d_evasion.py")
        if not os.path.exists(evasion_script):
            evasion_script = "phase_d_evasion.py"
            if not os.path.exists(evasion_script):
                evasion_script = "model_v2.0/phase_d_evasion.py"

        cmd = [
            sys.executable, evasion_script,
            "--data_path", args.data_path,
            "--splits_dir", args.splits_dir,
            "--freeze_dir", args.freeze_dir,
            "--output_csv", "outputs/replay_payloads.csv",
            "--generations", "150",
            "--num_seeds", "25"
        ]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
