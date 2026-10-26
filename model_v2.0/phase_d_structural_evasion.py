"""
Phase D Structural Evasion Round (Item 3):
Tests 100 SQLi + 100 Code Injection test positives against the frozen model
using structural transformations (not WAF-A-MoLE semantics mutations):
  - payload placement: query arg / URL path / POST body (CT-less)
  - body truncation padding: 600-char benign prefix to exploit 500-char body limit
  - token overflow padding: enough benign path prefix to exceed 256 tokens
  - header placement: Cookie / Referer / X-Forwarded-For (not serialized)
  - double URL-encoding, mixed case encoding, unicode escapes
  - UA replacement: 5 benign browser UAs plus the mutation

Outputs replay_payloads2.csv for CRS replay.
"""

import argparse
import copy
import json
import os
import random
import re
import sys
import urllib.parse
import numpy as np
import pandas as pd
from tqdm import tqdm

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

BENIGN_UAS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
]

GENERIC_UA = BENIGN_UAS[0]

# Benign padding text: Lorem ipsum (no attack tokens)
BENIGN_PAD_TEXT = ("lorem ipsum dolor sit amet consectetur adipiscing elit sed do "
                   "eiusmod tempor incididunt ut labore et dolore magna aliqua ut enim "
                   "ad minim veniam quis nostrud exercitation ullamco laboris nisi ut "
                   "aliquip ex ea commodo consequat duis aute irure dolor in reprehenderit "
                   "in voluptate velit esse cillum dolore eu fugiat nulla pariatur excepteur "
                   "sint occaecat cupidatat non proident sunt in culpa qui officia deserunt "
                   "mollit anim id est laborum ") * 10  # ~2600 chars


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


class ModelScorer:
    def __init__(self, model, tokenizer, thresholds, device):
        self.model = model
        self.tokenizer = tokenizer
        self.thresholds = thresholds
        self.device = device

    def score_text(self, text):
        """Returns (probs_array, firing_labels, is_attack)."""
        enc = self.tokenizer([text], truncation=True, max_length=MAX_LENGTH,
                             padding="max_length", return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            probs = torch.sigmoid(outputs["logits"])[0].cpu().numpy()
        firing = [LABEL_COLS[j] for j in range(len(LABEL_COLS))
                  if probs[j] >= self.thresholds.get(LABEL_COLS[j], 0.5)]
        is_attack = any(c != '000 - Normal' for c in firing)
        return probs, firing, is_attack

    def score_batch(self, texts, batch_size=128):
        all_probs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = self.tokenizer(batch, truncation=True, max_length=MAX_LENGTH,
                                 padding="max_length", return_tensors="pt")
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask)
                probs = torch.sigmoid(outputs["logits"]).cpu().numpy()
            all_probs.append(probs)
        return np.vstack(all_probs)

    def is_evaded(self, probs, target_col_idx):
        """Check if target label is below threshold AND no other attack label fires."""
        target_thresh = self.thresholds[LABEL_COLS[target_col_idx]]
        if probs[target_col_idx] >= target_thresh:
            return False
        for k in range(1, 13):
            if k != target_col_idx and probs[k] >= self.thresholds[LABEL_COLS[k]]:
                return False
        return True


def double_url_encode(payload):
    """Double URL-encode: e.g. ' -> %27 -> %2527."""
    first = urllib.parse.quote(payload, safe='')
    return urllib.parse.quote(first, safe='')


def mixed_case_encode(payload):
    """Mixed hex-case URL encoding: %2f and %2F randomly."""
    result = []
    for ch in payload:
        if ch.isalnum() or ch in '-_.~':
            result.append(ch)
        else:
            encoded = urllib.parse.quote(ch)
            mixed = ''.join(c.upper() if random.random() > 0.5 else c.lower() for c in encoded)
            result.append(mixed)
    return ''.join(result)


def unicode_escape(payload):
    """Replace ASCII characters with unicode escape sequences where possible."""
    result = []
    for ch in payload:
        if ch.isalpha() and random.random() > 0.5:
            result.append(f"%u{ord(ch):04X}")
        else:
            result.append(ch)
    return ''.join(result)


def resolve_path(p):
    for cand in [p, os.path.join("..", p),
                 os.path.join(os.path.dirname(__file__), "..", p),
                 os.path.join(os.path.dirname(__file__), p)]:
        if os.path.exists(cand):
            return os.path.abspath(cand)
    return p


def main():
    parser = argparse.ArgumentParser(description="Phase D Structural Evasion Round")
    parser.add_argument("--data_path", default="data/data_capec_multilabel.csv")
    parser.add_argument("--splits_dir", default="outputs/splits")
    parser.add_argument("--freeze_dir", default="outputs/frozen_v2_1b")
    parser.add_argument("--output_csv", default="outputs/replay_payloads2.csv")
    parser.add_argument("--num_seeds", type=int, default=100,
                        help="Number of seeds per attack class (default: 100)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, Available GPUs: {torch.cuda.device_count()}")

    args.freeze_dir = resolve_path(args.freeze_dir)
    args.splits_dir = resolve_path(args.splits_dir)
    args.data_path = resolve_path(args.data_path)

    # Resolve output path
    if not os.path.isabs(args.output_csv):
        if os.path.exists("../data") or os.path.exists("../outputs"):
            args.output_csv = os.path.abspath(os.path.join("..", args.output_csv))
        else:
            args.output_csv = os.path.abspath(args.output_csv)

    # Load thresholds
    thresh_path = os.path.join(args.freeze_dir, "thresholds.json")
    with open(thresh_path, "r") as f:
        thresholds = json.load(f)
    print(f"Loaded thresholds from {thresh_path}")

    # Load model
    weights_path = os.path.join(args.freeze_dir, "v2_1b_best.pt")
    if not os.path.exists(weights_path):
        weights_path = resolve_path("outputs/v2_1b/v2_1b_best.pt")
    print(f"Loading frozen model from {weights_path}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = MultiLabelWAFModel().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    scorer = ModelScorer(model, tokenizer, thresholds, device)

    # Load test data
    print(f"\nLoading dataset from {args.data_path}...")
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
    print(f"Test set: {len(df_test):,d} rows")

    # ══════════════════════════════════════════════════════════════════
    # SELECT SEEDS
    # ══════════════════════════════════════════════════════════════════
    attack_configs = [
        ("SQL Injection", "66 - SQL Injection", 5),   # col_idx 5
        ("Code Injection", "242 - Code Injection", 2), # col_idx 2
    ]

    all_records = []
    global_id = 1
    summary = {}

    for attack_name, label_col, col_idx in attack_configs:
        frozen_thresh = thresholds[label_col]
        pos_df = df_test[df_test[label_col] == 1].copy()
        print(f"\n{'='*100}")
        print(f"STRUCTURAL EVASION: {attack_name.upper()} ({len(pos_df):,d} total positives)")
        print(f"{'='*100}")

        # Score positives, pick confirmed-detected seeds
        cand_texts = []
        for _, r in pos_df.head(500).iterrows():
            cand_texts.append(serialize_request(
                r.get('request_http_method', 'GET'),
                r.get('request_http_request', '/'),
                GENERIC_UA,
                r.get('request_content_type', ''),
                r.get('request_body', '')
            ))

        cand_probs = scorer.score_batch(cand_texts)
        detected_mask = cand_probs[:, col_idx] >= frozen_thresh
        detected_indices = np.where(detected_mask)[0]

        n_needed = min(args.num_seeds, len(detected_indices))
        np.random.seed(42)
        chosen = np.random.choice(detected_indices, size=n_needed, replace=False)
        seed_df = pos_df.iloc[chosen].reset_index(drop=True)
        print(f"  Selected {n_needed} confirmed-detected seeds (T={frozen_thresh})")

        # Variant types and counters
        variant_stats = {}

        def record_variant(variant_type, seed_idx, orig_payload, method, path, ua, ct, body,
                           placement="standard", header_name="", header_value=""):
            nonlocal global_id
            text = serialize_request(method, path, ua, ct, body)
            probs, _, _ = scorer.score_text(text)
            target_score = float(probs[col_idx])
            evaded = scorer.is_evaded(probs, col_idx)

            if variant_type not in variant_stats:
                variant_stats[variant_type] = {"total": 0, "evaded": 0, "scores": []}
            variant_stats[variant_type]["total"] += 1
            variant_stats[variant_type]["scores"].append(target_score)
            if evaded:
                variant_stats[variant_type]["evaded"] += 1

            rec = {
                "id": global_id,
                "seed_id": f"{attack_name[:4]}_{seed_idx+1:03d}",
                "attack_type": attack_name,
                "variant_type": variant_type,
                "original_payload": orig_payload[:200],
                "mutated_payload": (body if placement == "body" else path)[:200],
                "http_method": method,
                "url_path_with_payload": path,
                "body": body,
                "content_type": ct,
                "model_score_original": round(float(cand_probs[chosen[seed_idx], col_idx]), 4),
                "model_score_mutated": round(target_score, 4),
                "model_evaded": evaded,
                "sqlparse_valid": True,
                "placement": placement,
                "header_name": header_name,
                "header_value": header_value[:200] if header_value else "",
            }
            all_records.append(rec)
            global_id += 1
            return evaded

        # Process each seed
        for sidx, s_row in tqdm(seed_df.iterrows(), total=len(seed_df), desc=f"  {attack_name} seeds"):
            orig_method = str(s_row.get('request_http_method', 'GET')).upper()
            orig_path = str(s_row.get('request_http_request', '/'))
            orig_body = str(s_row.get('request_body', '')) if pd.notna(s_row.get('request_body')) else ''
            orig_ct = str(s_row.get('request_content_type', '')) if pd.notna(s_row.get('request_content_type')) else ''

            # Determine payload
            if orig_body and len(orig_body.strip()) > 3:
                payload = orig_body.strip()
                orig_loc = "body"
            else:
                payload = orig_path.strip()
                orig_loc = "path"

            # --- VARIANT 1: Placement variations ---
            # (a) In query arg
            qa_path = f"/?id={urllib.parse.quote(payload, safe='')}"
            record_variant("placement_query_arg", sidx, payload,
                           "GET", qa_path, GENERIC_UA, "", "",
                           placement="path")

            # (b) In URL path
            p_path = payload if payload.startswith('/') else '/' + payload
            record_variant("placement_url_path", sidx, payload,
                           "GET", p_path, GENERIC_UA, "", "",
                           placement="path")

            # (c) In POST body (no Content-Type)
            record_variant("placement_post_body_no_ct", sidx, payload,
                           "POST", "/submit", GENERIC_UA, "", payload,
                           placement="body")

            # --- VARIANT 2: Body truncation padding ---
            # 600 chars of benign + payload; serializer truncates body at 500 chars
            padded_body = BENIGN_PAD_TEXT[:600] + payload
            record_variant("body_truncation_padding_600", sidx, payload,
                           "POST", "/submit", GENERIC_UA,
                           "application/x-www-form-urlencoded", padded_body,
                           placement="body")

            # --- VARIANT 3: Token overflow in path ---
            # Enough path prefix to exceed 256 tokens (~4 chars/token -> ~1100 chars)
            token_pad = "/benign/" + "a" * 1200 + "/"
            overflow_path = token_pad + payload
            record_variant("path_token_overflow_256", sidx, payload,
                           "GET", overflow_path, GENERIC_UA, "", "",
                           placement="path")

            # --- VARIANT 4: Header placement (NOT serialized -> expect evasion) ---
            for hdr_name in ["Cookie", "Referer", "X-Forwarded-For"]:
                record_variant(f"header_{hdr_name.lower()}", sidx, payload,
                               "GET", "/", GENERIC_UA, "", "",
                               placement="header",
                               header_name=hdr_name, header_value=payload)

            # --- VARIANT 5: Encoding transformations ---
            # Double URL-encoding
            dbl_encoded = double_url_encode(payload)
            if orig_loc == "path":
                record_variant("double_url_encoding", sidx, payload,
                               "GET", dbl_encoded if dbl_encoded.startswith('/') else '/' + dbl_encoded,
                               GENERIC_UA, "", "", placement="path")
            else:
                record_variant("double_url_encoding", sidx, payload,
                               "POST", "/submit", GENERIC_UA,
                               "application/x-www-form-urlencoded", dbl_encoded,
                               placement="body")

            # Mixed case encoding
            mc_encoded = mixed_case_encode(payload)
            if orig_loc == "path":
                record_variant("mixed_case_encoding", sidx, payload,
                               "GET", mc_encoded if mc_encoded.startswith('/') else '/' + mc_encoded,
                               GENERIC_UA, "", "", placement="path")
            else:
                record_variant("mixed_case_encoding", sidx, payload,
                               "POST", "/submit", GENERIC_UA,
                               "application/x-www-form-urlencoded", mc_encoded,
                               placement="body")

            # Unicode escapes
            uni_encoded = unicode_escape(payload)
            if orig_loc == "path":
                record_variant("unicode_escapes", sidx, payload,
                               "GET", uni_encoded if uni_encoded.startswith('/') else '/' + uni_encoded,
                               GENERIC_UA, "", "", placement="path")
            else:
                record_variant("unicode_escapes", sidx, payload,
                               "POST", "/submit", GENERIC_UA,
                               "application/x-www-form-urlencoded", uni_encoded,
                               placement="body")

            # --- VARIANT 6: UA replacement (5 benign UAs + payload in UA) ---
            for ua_idx, benign_ua in enumerate(BENIGN_UAS):
                if orig_loc == "body":
                    record_variant(f"ua_swap_{ua_idx}", sidx, payload,
                                   orig_method, orig_path, benign_ua, orig_ct, payload,
                                   placement="body")
                else:
                    record_variant(f"ua_swap_{ua_idx}", sidx, payload,
                                   "GET", orig_path, benign_ua, "", "",
                                   placement="path")

            # Payload injected INTO the UA string
            attack_ua = GENERIC_UA + " " + payload[:100]
            if orig_loc == "body":
                record_variant("ua_mutation_inject", sidx, payload,
                               orig_method, "/", attack_ua, "", "",
                               placement="header", header_name="User-Agent",
                               header_value=attack_ua)
            else:
                record_variant("ua_mutation_inject", sidx, payload,
                               "GET", "/", attack_ua, "", "",
                               placement="header", header_name="User-Agent",
                               header_value=attack_ua)

        # Print variant summary for this attack type
        print(f"\n  STRUCTURAL EVASION RESULTS: {attack_name.upper()}")
        print(f"  {'Variant Type':<35} {'Total':>6} {'Evaded':>8} {'Evasion %':>10} {'Mean Score':>11} {'Min Score':>10}")
        print("  " + "-" * 90)

        for vtype in sorted(variant_stats.keys()):
            vs = variant_stats[vtype]
            total = vs["total"]
            evaded = vs["evaded"]
            scores = vs["scores"]
            rate = f"{100.0 * evaded / total:.1f}%" if total > 0 else "N/A"
            mean_s = f"{np.mean(scores):.4f}" if scores else "N/A"
            min_s = f"{np.min(scores):.4f}" if scores else "N/A"
            print(f"  {vtype:<35} {total:>6} {evaded:>8} {rate:>10} {mean_s:>11} {min_s:>10}")

        summary[attack_name] = variant_stats
        print()

    # Save replay CSV
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    df_out = pd.DataFrame(all_records)
    df_out.to_csv(args.output_csv, index=False)
    print(f"\n[DONE] Saved {len(df_out):,d} structural evasion variants to {args.output_csv}")

    # Grand summary
    print("\n" + "=" * 100)
    print("GRAND SUMMARY: STRUCTURAL EVASION (ALL ATTACK TYPES)")
    print("=" * 100)
    for attack_name, vstats in summary.items():
        total_all = sum(v["total"] for v in vstats.values())
        evaded_all = sum(v["evaded"] for v in vstats.values())
        rate = f"{100.0 * evaded_all / total_all:.1f}%" if total_all > 0 else "N/A"
        print(f"\n  [{attack_name.upper()}]")
        print(f"    Total Variants Tested: {total_all:,d}")
        print(f"    Total Evasions:        {evaded_all:,d} ({rate})")

        # Group by category
        categories = {
            "Placement": [k for k in vstats if k.startswith("placement_")],
            "Body Padding/Truncation": [k for k in vstats if "truncation" in k or "overflow" in k],
            "Header Placement (unserialized)": [k for k in vstats if k.startswith("header_")],
            "Encoding": [k for k in vstats if k in ("double_url_encoding", "mixed_case_encoding", "unicode_escapes")],
            "UA Variations": [k for k in vstats if k.startswith("ua_")],
        }
        for cat, keys in categories.items():
            cat_total = sum(vstats[k]["total"] for k in keys if k in vstats)
            cat_evaded = sum(vstats[k]["evaded"] for k in keys if k in vstats)
            cat_rate = f"{100.0 * cat_evaded / cat_total:.1f}%" if cat_total > 0 else "N/A"
            print(f"      {cat:<40} {cat_evaded:>5}/{cat_total:<5} ({cat_rate})")

    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
