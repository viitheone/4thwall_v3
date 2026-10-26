"""
Phase D Hybrid Cost (Item 4):
Sample 5,000 random true-Normal test rows (stratified across templates, not
just the giant crawler template) and:
  1. Write them to outputs/normal_crs_sample.csv for CRS replay.
  2. Report model FPR at frozen thresholds (excl + incl 272).
  3. After CRS replay, report CRS FPR and union (hybrid) FPR.

Usage:
  Step A (on Kaggle or locally with data):
    python model_v2.0/phase_d_hybrid_cost.py --mode generate \
        --data_path data/data_capec_multilabel.csv \
        --splits_dir outputs/splits \
        --freeze_dir outputs/frozen_v2_1b \
        --output_csv outputs/normal_crs_sample.csv

  Step B (locally, after CRS replay):
    python model_v2.0/phase_d_hybrid_cost.py --mode evaluate \
        --crs_csv outputs/normal_crs_results.csv \
        --model_csv outputs/normal_model_scores.csv
"""

import argparse
import json
import os
import sys
import random
import urllib.parse
import numpy as np
import pandas as pd

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

MAX_BODY_CHARS = 500
GENERIC_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
              "AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/120.0.0.0 Safari/537.36")


def serialize_request(method="GET", path="/", ua="", ct="", body="", max_body_chars=MAX_BODY_CHARS):
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


def resolve_path(p):
    for cand in [p, os.path.join("..", p),
                 os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", p),
                 os.path.join(os.path.dirname(os.path.abspath(__file__)), p)]:
        if os.path.exists(cand):
            return os.path.abspath(cand)
    return p


def mode_generate(args):
    """Generate 5000 Normal test rows, score with model, write CSVs."""
    import torch
    import torch.nn as nn
    from transformers import AutoTokenizer, AutoModel
    from tqdm import tqdm

    BASE_MODEL = "distilbert-base-uncased"
    MAX_LENGTH = 256

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    args.freeze_dir = resolve_path(args.freeze_dir)
    args.splits_dir = resolve_path(args.splits_dir)
    args.data_path = resolve_path(args.data_path)

    if not os.path.isabs(args.output_csv):
        if os.path.exists("../data") or os.path.exists("../outputs"):
            args.output_csv = os.path.abspath(os.path.join("..", args.output_csv))
        else:
            args.output_csv = os.path.abspath(args.output_csv)

    # Load thresholds
    thresh_path = os.path.join(args.freeze_dir, "thresholds.json")
    with open(thresh_path, "r") as f:
        thresholds = json.load(f)

    # Load model
    weights_path = os.path.join(args.freeze_dir, "v2_1b_best.pt")
    if not os.path.exists(weights_path):
        weights_path = resolve_path("outputs/v2_1b/v2_1b_best.pt")
    print(f"Loading model from {weights_path}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = MultiLabelWAFModel().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # Load dataset and test split
    print(f"Loading data from {args.data_path}...")
    df = pd.read_csv(args.data_path, low_memory=False)
    if '248 - Command Injection' in df.columns and '88 - OS Command Injection' in df.columns:
        df['88 - OS Command Injection'] = (
            df['88 - OS Command Injection'].fillna(0).astype(int) |
            df['248 - Command Injection'].fillna(0).astype(int)
        ).astype(int)
        df = df.drop(columns=['248 - Command Injection'])
    df = df.dropna(subset=['request_http_request']).reset_index(drop=True)

    test_idx = np.load(os.path.join(args.splits_dir, "grouped_test_idx.npy"))
    df_test = df.iloc[test_idx].copy().reset_index(drop=True)

    # Filter true-Normal rows (label 000=1, all attack labels=0)
    attack_cols = [c for c in LABEL_COLS if c != '000 - Normal']
    is_normal = (df_test['000 - Normal'] == 1)
    for ac in attack_cols:
        if ac in df_test.columns:
            is_normal = is_normal & (df_test[ac] == 0)
    normal_df = df_test[is_normal].copy().reset_index(drop=True)
    print(f"True-Normal test rows: {len(normal_df):,d}")

    # Template-stratified sampling: create template key and sample across templates
    def make_template_key(row):
        method = str(row.get('request_http_method', '')).upper()
        path = str(row.get('request_http_request', ''))
        path_norm = path.split('?')[0].lower().strip()
        ua = str(row.get('request_http_headers_useragent', ''))[:60]
        return f"{method}|{path_norm}|{ua}"

    print("Computing template keys for stratified sampling...")
    normal_df['_template_key'] = normal_df.apply(make_template_key, axis=1)
    unique_templates = normal_df['_template_key'].unique()
    n_templates = len(unique_templates)
    print(f"  Unique templates in Normal test rows: {n_templates:,d}")

    # Stratified sample: up to N rows per template, shuffle, take 5000
    N_SAMPLE = args.num_rows
    random.seed(42)
    np.random.seed(42)

    if n_templates >= N_SAMPLE:
        # 1 row per template, sample 5000 templates
        chosen_templates = np.random.choice(unique_templates, size=N_SAMPLE, replace=False)
        sampled = normal_df.groupby('_template_key').apply(
            lambda x: x.sample(1, random_state=42) if x.name in chosen_templates else x.head(0)
        ).reset_index(drop=True)
    else:
        # Spread across all templates
        per_template = max(1, N_SAMPLE // n_templates)
        remainder = N_SAMPLE - per_template * n_templates
        sampled_parts = []
        for tk in unique_templates:
            tk_rows = normal_df[normal_df['_template_key'] == tk]
            n_take = min(per_template, len(tk_rows))
            sampled_parts.append(tk_rows.sample(n=n_take, random_state=42))
        sampled = pd.concat(sampled_parts, ignore_index=True)
        if len(sampled) < N_SAMPLE:
            leftover = normal_df[~normal_df.index.isin(sampled.index)]
            extra = leftover.sample(n=min(N_SAMPLE - len(sampled), len(leftover)), random_state=42)
            sampled = pd.concat([sampled, extra], ignore_index=True)
        sampled = sampled.head(N_SAMPLE)

    sampled = sampled.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"  Sampled {len(sampled):,d} rows across {sampled['_template_key'].nunique():,d} unique templates")

    # Serialize and score with model
    print("\nScoring sampled Normal rows with frozen model...")
    texts = []
    replay_records = []

    for idx, row in sampled.iterrows():
        method = str(row.get('request_http_method', 'GET')).upper()
        path = str(row.get('request_http_request', '/'))
        ua_col = None
        for c in ['request_http_headers_useragent', 'request_http_headers_UserAgent']:
            if c in row.index:
                ua_col = c
                break
        ua = str(row.get(ua_col, GENERIC_UA)) if ua_col else GENERIC_UA
        if pd.isna(ua) or str(ua).lower() in ('nan', 'none', ''):
            ua = GENERIC_UA
        ct = str(row.get('request_content_type', '')) if pd.notna(row.get('request_content_type')) else ''
        body_raw = str(row.get('request_body', '')) if pd.notna(row.get('request_body')) else ''

        text = serialize_request(method, path, ua, ct, body_raw)
        texts.append(text)

        replay_records.append({
            "id": idx + 1,
            "http_method": method,
            "url_path_with_payload": path,
            "body": body_raw,
            "content_type": ct,
            "user_agent": ua,
        })

    # Batch score
    batch_size = 128
    all_probs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="  Scoring"):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, truncation=True, max_length=MAX_LENGTH,
                        padding="max_length", return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            probs = torch.sigmoid(outputs["logits"]).cpu().numpy()
        all_probs.append(probs)
    all_probs = np.vstack(all_probs)

    # Compute model FP stats
    print("\n" + "=" * 100)
    print("MODEL v2.1b FPR ON 5,000 SAMPLED TRUE-NORMAL TEST ROWS (FROZEN THRESHOLDS)")
    print("=" * 100)

    any_attack_fp = np.zeros(len(all_probs), dtype=bool)
    any_attack_excl_272 = np.zeros(len(all_probs), dtype=bool)

    print(f"\n  {'Label':<45} {'Threshold':>10} {'FP Count':>10} {'FPR %':>10}")
    print("  " + "-" * 80)

    for j in range(len(LABEL_COLS)):
        label = LABEL_COLS[j]
        if label == '000 - Normal':
            continue
        thresh = thresholds.get(label, 0.5)
        fires = all_probs[:, j] >= thresh
        fp_count = int(fires.sum())
        fpr = 100.0 * fp_count / len(all_probs)
        print(f"  {label:<45} {thresh:>10.2f} {fp_count:>10} {fpr:>9.2f}%")

        any_attack_fp |= fires
        if label != '272 - Protocol Manipulation':
            any_attack_excl_272 |= fires

    total_fp = int(any_attack_fp.sum())
    total_fp_excl = int(any_attack_excl_272.sum())
    print()
    print(f"  Overall FP (any attack label fires):             {total_fp:>6} / {len(all_probs)} ({100.0*total_fp/len(all_probs):.2f}%)")
    print(f"  Overall FP (excl 272 - Protocol Manipulation):   {total_fp_excl:>6} / {len(all_probs)} ({100.0*total_fp_excl/len(all_probs):.2f}%)")

    # Add model scores to replay records
    for i, rec in enumerate(replay_records):
        rec["model_any_attack_fp"] = bool(any_attack_fp[i])
        rec["model_any_attack_excl272_fp"] = bool(any_attack_excl_272[i])
        for j, label in enumerate(LABEL_COLS):
            rec[f"model_prob_{label}"] = round(float(all_probs[i, j]), 4)

    # Save CRS replay CSV (same schema as replay_crs.py expects)
    df_replay = pd.DataFrame(replay_records)
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    df_replay.to_csv(args.output_csv, index=False)
    print(f"\n  Saved {len(df_replay):,d} normal rows for CRS replay to {args.output_csv}")

    # Also save model scores separately for the evaluate step
    model_scores_path = args.output_csv.replace(".csv", "_model_scores.csv")
    df_replay.to_csv(model_scores_path, index=False)
    print(f"  Saved model scores to {model_scores_path}")
    print("=" * 100)


def mode_evaluate(args):
    """After CRS replay: combine CRS + model results and report hybrid FPR."""
    crs_df = pd.read_csv(args.crs_csv)
    model_df = pd.read_csv(args.model_csv)

    n = len(crs_df)
    print(f"Loaded {n:,d} CRS-replayed Normal rows from {args.crs_csv}")
    print(f"Loaded {len(model_df):,d} model-scored Normal rows from {args.model_csv}")

    # CRS FP = row blocked by CRS (403) when it's actually benign
    if 'crs_blocked' in crs_df.columns:
        crs_fp = crs_df['crs_blocked'].astype(bool)
    elif 'http_status' in crs_df.columns:
        crs_fp = (crs_df['http_status'] == 403)
    else:
        print("[ERROR] No crs_blocked or http_status column in CRS results.")
        sys.exit(1)

    # Model FP columns
    model_fp_all = model_df['model_any_attack_fp'].astype(bool) if 'model_any_attack_fp' in model_df.columns else pd.Series([False]*n)
    model_fp_excl272 = model_df['model_any_attack_excl272_fp'].astype(bool) if 'model_any_attack_excl272_fp' in model_df.columns else pd.Series([False]*n)

    crs_fp_count = int(crs_fp.sum())
    model_fp_all_count = int(model_fp_all.sum())
    model_fp_excl_count = int(model_fp_excl272.sum())

    # Union FP (including 272): CRS blocks OR model fires any attack label
    union_fp_all = crs_fp | model_fp_all
    union_fp_all_count = int(union_fp_all.sum())

    # Union FP (excluding 272): CRS blocks OR model fires any non-272 attack label
    union_fp_excl = crs_fp | model_fp_excl272
    union_fp_excl_count = int(union_fp_excl.sum())

    print("\n" + "=" * 100)
    print("HYBRID WAF FPR ON 5,000 SAMPLED TRUE-NORMAL TEST ROWS")
    print("=" * 100)
    print(f"\n  {'Metric':<55} {'FP Count':>10} {'FPR %':>10}")
    print("  " + "-" * 80)
    print(f"  {'CRS FPR (blocked as 403 on benign traffic)':<55} {crs_fp_count:>10} {100.0*crs_fp_count/n:>9.2f}%")
    print(f"  {'Model FPR (any attack label, incl 272)':<55} {model_fp_all_count:>10} {100.0*model_fp_all_count/n:>9.2f}%")
    print(f"  {'Model FPR (excl 272 - Protocol Manipulation)':<55} {model_fp_excl_count:>10} {100.0*model_fp_excl_count/n:>9.2f}%")
    print(f"  {'Union FPR (CRS OR Model, incl 272)':<55} {union_fp_all_count:>10} {100.0*union_fp_all_count/n:>9.2f}%")
    print(f"  {'Union FPR (CRS OR Model, excl 272)':<55} {union_fp_excl_count:>10} {100.0*union_fp_excl_count/n:>9.2f}%")

    # Breakdown
    both = int((crs_fp & model_fp_all).sum())
    crs_only = int((crs_fp & ~model_fp_all).sum())
    model_only = int((~crs_fp & model_fp_all).sum())
    neither = int((~crs_fp & ~model_fp_all).sum())

    print(f"\n  BREAKDOWN (incl 272):")
    print(f"    Both CRS and Model fire FP:       {both:>6}")
    print(f"    CRS only fires FP:                {crs_only:>6}")
    print(f"    Model only fires FP:              {model_only:>6}")
    print(f"    Neither fires FP (true negative):  {neither:>6}")

    print("\n" + "=" * 100)


def main():
    parser = argparse.ArgumentParser(description="Phase D Hybrid Cost Analysis")
    parser.add_argument("--mode", choices=["generate", "evaluate"], required=True,
                        help="'generate' = sample & score Normal rows; 'evaluate' = combine CRS + model results")

    # Generate mode args
    parser.add_argument("--data_path", default="data/data_capec_multilabel.csv")
    parser.add_argument("--splits_dir", default="outputs/splits")
    parser.add_argument("--freeze_dir", default="outputs/frozen_v2_1b")
    parser.add_argument("--output_csv", default="outputs/normal_crs_sample.csv")
    parser.add_argument("--num_rows", type=int, default=5000)

    # Evaluate mode args
    parser.add_argument("--crs_csv", default="outputs/normal_crs_results.csv")
    parser.add_argument("--model_csv", default="outputs/normal_crs_sample_model_scores.csv")

    args = parser.parse_args()

    if args.mode == "generate":
        mode_generate(args)
    else:
        mode_evaluate(args)


if __name__ == "__main__":
    main()
