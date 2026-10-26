"""
Phase E1.5 Item 5: FPR Analysis on Three Benign Sets
1. Set (i): Template-Stratified (5,000 requests)
2. Set (ii): Traffic-Weighted (5,000 requests)
3. Set (iii): WordPress Browsing / Crawler Traffic (normal browser UAs, articles, search ?s=, forms)
Scores all three with Frozen Model v2.1b (frozen thresholds).
Replays Set (iii) against CRS at PL1, threshold 5.
Reports Model, CRS, and Union FPR across all three sets.
Lists the top benign templates behind CRS FPs.
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
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from collections import Counter

BASE_MODEL = "distilbert-base-uncased"
MAX_LENGTH = 256
GENERIC_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
BASE_URL = "http://localhost:80"

LABEL_COLS = [
    '000 - Normal', '272 - Protocol Manipulation', '242 - Code Injection',
    '88 - OS Command Injection', '126 - Path Traversal', '66 - SQL Injection',
    '16 - Dictionary-based Password Attack', '310 - Scanning for Vulnerable Software',
    '153 - Input Data Manipulation', '274 - HTTP Verb Tampering',
    '194 - Fake the Source of Data', '34 - HTTP Response Splitting',
    '33 - HTTP Request Smuggling'
]

class MultiLabelWAFModel(nn.Module):
    def __init__(self, model_name=BASE_MODEL, num_labels=13, dropout=0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.bert.config.hidden_size, num_labels))
    def forward(self, input_ids, attention_mask):
        cls = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        return {'logits': self.classifier(cls)}

def serialize_request(method='GET', path='/', ua='', ct='', body=''):
    parts = []
    if method: parts.append(f'[METHOD] {str(method).strip()}')
    if path: parts.append(f'[PATH] {str(path).strip()}')
    if ua: parts.append(f'[UA] {str(ua).strip()}')
    if ct: parts.append(f'[CT] {str(ct).strip()}')
    if body: parts.append(f'[BODY] {str(body).strip()[:500]}')
    return ' '.join(parts)

def main():
    print("=" * 110)
    print("PHASE E1.5 ITEM 5: FPR EVALUATION (MODEL, CRS, UNION ON 3 BENIGN DATASETS)")
    print("=" * 110)

    # 1. Load Model & Thresholds
    print("Loading Frozen Model v2.1b...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = MultiLabelWAFModel()
    model.load_state_dict(torch.load("model_v2.0/outputs/v2_1b/v2_1b_best.pt", map_location="cpu"))
    model.eval()

    with open("outputs/frozen_v2_1b/thresholds.json") as f:
        thresholds = json.load(f)

    def score_df(df, path_col='url_path', method_col='method', body_col='body', ua_col=None, ct_col=None):
        texts = []
        for _, r in df.iterrows():
            m = str(r.get(method_col, 'GET')).upper()
            p = str(r.get(path_col, '/'))
            b = str(r.get(body_col, '')) if pd.notna(r.get(body_col)) else ''
            ua = str(r.get(ua_col, GENERIC_UA)) if ua_col and pd.notna(r.get(ua_col)) else GENERIC_UA
            ct = str(r.get(ct_col, '')) if ct_col and pd.notna(r.get(ct_col)) else ''
            texts.append(serialize_request(m, p, ua, ct, b))

        batch_size = 64
        all_probs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tokenizer(batch, truncation=True, max_length=MAX_LENGTH, padding="max_length", return_tensors="pt")
            with torch.no_grad():
                out = model(enc["input_ids"], enc["attention_mask"])
                probs = torch.sigmoid(out["logits"]).cpu().numpy()
                all_probs.append(probs)
        probs = np.vstack(all_probs)

        # FP check: any attack label >= threshold
        any_fp = np.zeros(len(probs), dtype=bool)
        any_fp_excl272 = np.zeros(len(probs), dtype=bool)
        for j, col_name in enumerate(LABEL_COLS):
            if col_name == '000 - Normal':
                continue
            thresh = thresholds[col_name]
            fired = probs[:, j] >= thresh
            any_fp |= fired
            if col_name != '272 - Protocol Manipulation':
                any_fp_excl272 |= fired

        return any_fp, any_fp_excl272

    # 2. Score Set (i): Template-Stratified (5,000 rows)
    print("\n--- Scoring Set (i): Template-Stratified (5,000 rows) ---")
    df_strat = pd.read_csv("outputs/phase_e1_crs_stratified_results.csv")
    strat_model_all, strat_model_excl = score_df(df_strat)
    df_strat['model_fp_all'] = strat_model_all
    df_strat['model_fp_excl'] = strat_model_excl
    df_strat['crs_fp'] = df_strat['crs_blocked']
    df_strat['union_fp_all'] = df_strat['crs_fp'] | df_strat['model_fp_all']
    df_strat['union_fp_excl'] = df_strat['crs_fp'] | df_strat['model_fp_excl']

    # 3. Score Set (ii): Traffic-Weighted (5,000 rows)
    print("--- Scoring Set (ii): Traffic-Weighted (5,000 rows) ---")
    df_weight = pd.read_csv("outputs/phase_e1_crs_weighted_results.csv")
    weight_model_all, weight_model_excl = score_df(df_weight)
    df_weight['model_fp_all'] = weight_model_all
    df_weight['model_fp_excl'] = weight_model_excl
    df_weight['crs_fp'] = df_weight['crs_blocked']
    df_weight['union_fp_all'] = df_weight['crs_fp'] | df_weight['model_fp_all']
    df_weight['union_fp_excl'] = df_weight['crs_fp'] | df_weight['model_fp_excl']

    # 4. Generate & Score Set (iii): WordPress Browsing & Search Traffic (3,000 rows)
    print("--- Preparing & Replaying Set (iii): WordPress Browsing & Search Traffic (3,000 rows) ---")
    df_capec = pd.read_csv("model_v2.0/data/data_capec_multilabel.csv", low_memory=False)
    df_normal = df_capec[df_capec['000 - Normal'] == 1].copy()
    for col in LABEL_COLS:
        if col != '000 - Normal' and col in df_normal.columns:
            df_normal = df_normal[df_normal[col] == 0]

    # Filter to WordPress browsing (contains /blog/, /wp-, search ?s=, feed, etc. with normal desktop/mobile browser UAs)
    wp_mask = df_normal['request_http_request'].str.contains(r'/blog/|wp-|feed|xmlrpc|\?s=', case=False, na=False)
    wp_df = df_normal[wp_mask].copy()
    print(f"  Found {len(wp_df):,d} WordPress true-normal rows.")

    # Sample 3,000 diverse WordPress requests
    random.seed(42)
    wp_sample = wp_df.sample(n=min(3000, len(wp_df)), random_state=42).reset_index(drop=True)

    # Replay Set (iii) against CRS
    print("  Replaying 3,000 WordPress requests against CRS @ PL1, threshold=5...")
    session = requests.Session()
    session.headers.update({"User-Agent": GENERIC_UA})

    wp_crs_results = []
    for idx, r in wp_sample.iterrows():
        method = str(r.get('request_http_method', 'GET')).upper()
        p = str(r.get('request_http_request', '/'))
        if not p.startswith('/'):
            p = '/' + p
        b = str(r.get('request_body', '')) if pd.notna(r.get('request_body')) else ''
        ua = str(r.get('request_user_agent', GENERIC_UA)) if pd.notna(r.get('request_user_agent')) else GENERIC_UA
        ct = str(r.get('request_content_type', '')) if pd.notna(r.get('request_content_type')) else ''
        host = str(r.get('request_host', 'localhost')) if pd.notna(r.get('request_host')) else 'localhost'
        headers = {"User-Agent": ua, "Host": host}
        if ct: headers["Content-Type"] = ct

        target_url = urllib.parse.urljoin(BASE_URL, p)
        try:
            if method == 'POST':
                resp = session.post(target_url, data=b.encode('utf-8', errors='ignore'), headers=headers, timeout=5.0, allow_redirects=False)
            else:
                resp = session.get(target_url, headers=headers, timeout=5.0, allow_redirects=False)
            status = resp.status_code
        except Exception:
            status = 0

        wp_crs_results.append({
            "url_path": p,
            "method": method,
            "body": b,
            "user_agent": ua,
            "content_type": ct,
            "http_status": status,
            "crs_blocked": status == 403,
        })
        time.sleep(0.01)

    df_wp = pd.DataFrame(wp_crs_results)
    wp_model_all, wp_model_excl = score_df(df_wp, ua_col='user_agent', ct_col='content_type')
    df_wp['model_fp_all'] = wp_model_all
    df_wp['model_fp_excl'] = wp_model_excl
    df_wp['crs_fp'] = df_wp['crs_blocked']
    df_wp['union_fp_all'] = df_wp['crs_fp'] | df_wp['model_fp_all']
    df_wp['union_fp_excl'] = df_wp['crs_fp'] | df_wp['model_fp_excl']

    df_wp.to_csv("outputs/phase_e1_5_wordpress_results.csv", index=False)

    # Summary Table
    def report_set(name, d):
        n = len(d)
        crs_fp = int(d['crs_fp'].sum())
        mod_all = int(d['model_fp_all'].sum())
        mod_excl = int(d['model_fp_excl'].sum())
        union_all = int(d['union_fp_all'].sum())
        union_excl = int(d['union_fp_excl'].sum())

        both = int((d['crs_fp'] & d['model_fp_all']).sum())
        crs_only = int((d['crs_fp'] & ~d['model_fp_all']).sum())
        mod_only = int((~d['crs_fp'] & d['model_fp_all']).sum())
        neither = int((~d['crs_fp'] & ~d['model_fp_all']).sum())

        print(f"\n==========================================================================================")
        print(f" BENIGN SET: {name} (N = {n:,d})")
        print(f"==========================================================================================")
        print(f"  {'Defense Configuration':<45} {'FP Count':>10} {'FPR %':>10}")
        print("  " + "-" * 70)
        print(f"  {'ModSecurity CRS (PL1, Threshold 5)':<45} {crs_fp:>10} {100.0*crs_fp/n:>9.2f}%")
        print(f"  {'Frozen Model v2.1b (All Labels)':<45} {mod_all:>10} {100.0*mod_all/n:>9.2f}%")
        print(f"  {'Frozen Model v2.1b (Excl Label 272)':<45} {mod_excl:>10} {100.0*mod_excl/n:>9.2f}%")
        print(f"  {'Union WAF (CRS OR Model, Incl 272)':<45} {union_all:>10} {100.0*union_all/n:>9.2f}%")
        print(f"  {'Union WAF (CRS OR Model, Excl 272)':<45} {union_excl:>10} {100.0*union_excl/n:>9.2f}%")
        print("  " + "-" * 70)
        print(f"  Overlap Breakdown (with Label 272):")
        print(f"    - CRS Only Fired False Alarm:          {crs_only:>6} ({100.0*crs_only/n:>5.2f}%)")
        print(f"    - Model Only Fired False Alarm:        {mod_only:>6} ({100.0*mod_only/n:>5.2f}%)")
        print(f"    - Both CRS and Model Fired FP:         {both:>6} ({100.0*both/n:>5.2f}%)")
        print(f"    - Neither Fired (True Negatives):      {neither:>6} ({100.0*neither/n:>5.2f}%)")

    report_set("(i) Template-Stratified Benign Test Sample", df_strat)
    report_set("(ii) Traffic-Weighted Benign Test Sample", df_weight)
    report_set("(iii) WordPress Browsing, Search & Form Traffic", df_wp)

    # List Benign Templates behind CRS FPs across sets
    print("\n" + "=" * 110)
    print("TOP BENIGN URL TEMPLATES BEHIND CRS FALSE POSITIVES")
    print("=" * 110)

    for sname, d in [("Template-Stratified", df_strat), ("Traffic-Weighted", df_weight), ("WordPress Set", df_wp)]:
        fps = d[d['crs_fp'] == True]
        print(f"\n>>> Top CRS FP Templates in {sname} ({len(fps)} total FPs):")
        
        def template_path(p):
            p = str(p).split('?')[0]
            # Replace numeric IDs
            p = '/'.join(['<id>' if part.isdigit() else part for part in p.split('/')])
            return p

        templates = fps['url_path'].apply(template_path)
        counts = Counter(templates).most_common(5)
        for t, c in counts:
            print(f"    [{c:>3} hits] {t}")

    print("\n" + "=" * 110)

if __name__ == "__main__":
    main()
