"""
Phase E1 Item (c): Benign Replay with Original Headers at PL1 Threshold 5
1. Generate two 5,000-row benign test samples from test split:
   (i) Template-stratified sample (with original Accept, Connection, Host headers)
   (ii) Traffic-weighted random sample (with original Accept, Connection, Host headers)
2. Replay both against CRS (now at PL1, ANOMALY_INBOUND=5).
3. Report FPR and status distribution for both.
"""

import json
import os
import random
import sys
import time
import urllib.parse
import numpy as np
import pandas as pd
import requests

DATA_PATH = "model_v2.0/data/data_capec_multilabel.csv"
SPLITS_DIR = "model_v2.0/outputs/splits"
BASE_URL = "http://localhost:80"
RATE_LIMIT = 40.0
GENERIC_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

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

def load_normal_test_rows():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    if '248 - Command Injection' in df.columns and '88 - OS Command Injection' in df.columns:
        df['88 - OS Command Injection'] = (
            df['88 - OS Command Injection'].fillna(0).astype(int) |
            df['248 - Command Injection'].fillna(0).astype(int)
        ).astype(int)
        df = df.drop(columns=['248 - Command Injection'])
    df = df.dropna(subset=['request_http_request']).reset_index(drop=True)

    test_idx = np.load(os.path.join(SPLITS_DIR, "grouped_test_idx.npy"))
    df_test = df.iloc[test_idx].copy().reset_index(drop=True)

    attack_cols = [c for c in LABEL_COLS if c != '000 - Normal']
    is_normal = (df_test['000 - Normal'] == 1)
    for ac in attack_cols:
        if ac in df_test.columns:
            is_normal = is_normal & (df_test[ac] == 0)
    normal_df = df_test[is_normal].copy().reset_index(drop=True)
    print(f"Total True-Normal test rows: {len(normal_df):,d}")
    return normal_df

def make_template_key(row):
    method = str(row.get('request_http_method', '')).upper()
    path = str(row.get('request_http_request', ''))
    path_norm = path.split('?')[0].lower().strip()
    ua = str(row.get('request_user_agent', ''))[:60]
    return f"{method}|{path_norm}|{ua}"

def sample_stratified(normal_df, n_sample=5000):
    print("\nCreating (i) Template-Stratified Sample (5,000 rows)...")
    normal_df['_template_key'] = normal_df.apply(make_template_key, axis=1)
    unique_templates = normal_df['_template_key'].unique()
    n_templates = len(unique_templates)
    print(f"  Unique templates: {n_templates:,d}")

    random.seed(42)
    np.random.seed(42)

    if n_templates >= n_sample:
        chosen_templates = np.random.choice(unique_templates, size=n_sample, replace=False)
        sampled = normal_df.groupby('_template_key').apply(
            lambda x: x.sample(1, random_state=42) if x.name in chosen_templates else x.head(0)
        ).reset_index(drop=True)
    else:
        per_template = max(1, n_sample // n_templates)
        sampled_parts = []
        for tk in unique_templates:
            tk_rows = normal_df[normal_df['_template_key'] == tk]
            n_take = min(per_template, len(tk_rows))
            sampled_parts.append(tk_rows.sample(n=n_take, random_state=42))
        sampled = pd.concat(sampled_parts, ignore_index=True)
        if len(sampled) < n_sample:
            leftover = normal_df[~normal_df.index.isin(sampled.index)]
            extra = leftover.sample(n=min(n_sample - len(sampled), len(leftover)), random_state=42)
            sampled = pd.concat([sampled, extra], ignore_index=True)
        sampled = sampled.head(n_sample)

    sampled = sampled.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"  Sampled {len(sampled):,d} stratified rows.")
    return sampled

def sample_traffic_weighted(normal_df, n_sample=5000):
    print("\nCreating (ii) Traffic-Weighted Random Sample (5,000 rows)...")
    sampled = normal_df.sample(n=n_sample, random_state=42).reset_index(drop=True)
    print(f"  Sampled {len(sampled):,d} traffic-weighted rows.")
    return sampled

def replay_sample(df, sample_name, out_csv):
    print(f"\n" + "=" * 90)
    print(f"REPLAYING: {sample_name} (5,000 rows against CRS @ PL1, threshold=5)")
    print("=" * 90)

    min_interval = 1.0 / RATE_LIMIT
    session = requests.Session()
    results = []

    start_time = time.time()
    last_req_time = 0.0

    for idx, row in df.iterrows():
        method = str(row.get('request_http_method', 'GET')).upper()
        url_path = str(row.get('request_http_request', '/'))
        if not url_path.startswith('/'):
            url_path = '/' + url_path

        body = row.get('request_body', '')
        body = '' if pd.isna(body) else str(body)

        # Build headers with original dataset values where available
        headers = {}

        ua = row.get('request_user_agent', '')
        headers['User-Agent'] = str(ua).strip() if pd.notna(ua) and str(ua).strip() and str(ua).lower() != 'nan' else GENERIC_UA

        ct = row.get('request_content_type', '')
        if pd.notna(ct) and str(ct).strip() and str(ct).lower() != 'nan':
            headers['Content-Type'] = str(ct).strip()

        accept = row.get('request_accept', '')
        if pd.notna(accept) and str(accept).strip() and str(accept).lower() != 'nan':
            headers['Accept'] = str(accept).strip()

        conn = row.get('request_connection', '')
        if pd.notna(conn) and str(conn).strip() and str(conn).lower() != 'nan':
            headers['Connection'] = str(conn).strip()

        host = row.get('request_host', '')
        if pd.notna(host) and str(host).strip() and str(host).lower() != 'nan':
            headers['Host'] = str(host).strip()
        else:
            headers['Host'] = 'localhost'

        # Rate limit
        now = time.time()
        elapsed = now - last_req_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

        target_url = urllib.parse.urljoin(BASE_URL, url_path)
        status_code = None
        error_msg = ""
        last_req_time = time.time()

        try:
            if method == 'POST':
                resp = session.post(target_url, data=body.encode('utf-8', errors='ignore'), headers=headers,
                                    timeout=5.0, allow_redirects=False)
            else:
                resp = session.get(target_url, headers=headers, timeout=5.0, allow_redirects=False)
            status_code = resp.status_code
        except requests.exceptions.RequestException as e:
            error_msg = str(e)
            status_code = 0

        is_blocked = (status_code == 403)
        is_evaded = (status_code != 403 and status_code != 0)

        results.append({
            "id": idx + 1,
            "method": method,
            "url_path": url_path,
            "body": body[:100],
            "headers_sent": str(headers),
            "http_status": status_code,
            "crs_blocked": is_blocked,
            "crs_evaded": is_evaded,
            "error": error_msg,
        })

        if (idx + 1) % 500 == 0 or (idx + 1) == len(df):
            rate = (idx + 1) / (time.time() - start_time)
            print(f"  [{idx+1:,d}/{len(df):,d}] Last: {status_code} | Speed: {rate:.1f} req/s")

    out_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"\n[DONE] Saved replay results to {out_csv}")

    n_tot = len(out_df)
    n_blk = int(out_df['crs_blocked'].sum())
    fpr = 100.0 * n_blk / n_tot
    print(f"  Sample: {sample_name}")
    print(f"  Total Requests:  {n_tot:,d}")
    print(f"  CRS Blocked (403 - FP): {n_blk:,d} ({fpr:.2f}%)")
    print(f"  CRS Passed (non-403):   {int(out_df['crs_evaded'].sum()):,d} ({100.0-fpr:.2f}%)")
    return fpr, n_blk, n_tot

def main():
    normal_df = load_normal_test_rows()

    # 1. Template-Stratified Sample
    strat_df = sample_stratified(normal_df, 5000)
    strat_csv = "outputs/benign_sample_stratified_headers.csv"
    strat_df.to_csv(strat_csv, index=False)

    # 2. Traffic-Weighted Sample
    weighted_df = sample_traffic_weighted(normal_df, 5000)
    weighted_csv = "outputs/benign_sample_traffic_weighted_headers.csv"
    weighted_df.to_csv(weighted_csv, index=False)

    # Replay (i)
    fpr_strat, blk_strat, tot_strat = replay_sample(
        strat_df,
        "Template-Stratified (5,000 requests, with original headers @ PL1 Threshold 5)",
        "outputs/phase_e1_crs_stratified_results.csv"
    )

    # Replay (ii)
    fpr_weight, blk_weight, tot_weight = replay_sample(
        weighted_df,
        "Traffic-Weighted (5,000 requests, with original headers @ PL1 Threshold 5)",
        "outputs/phase_e1_crs_weighted_results.csv"
    )

    print("\n" + "=" * 90)
    print("PHASE E1 (c) BENIGN FPR REPLAY SUMMARY (PL1, DEFAULT THRESHOLD 5, ORIGINAL HEADERS)")
    print("=" * 90)
    print(f"  (i) Template-Stratified Sample:    {blk_strat:,d} / {tot_strat:,d} ({fpr_strat:.2f}% FPR)")
    print(f"  (ii) Traffic-Weighted Random Sample: {blk_weight:,d} / {tot_weight:,d} ({fpr_weight:.2f}% FPR)")
    print("=" * 90)

if __name__ == "__main__":
    main()
