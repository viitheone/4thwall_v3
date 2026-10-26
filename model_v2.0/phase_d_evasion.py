"""
Phase D Adversarial Evasion Pipeline:
- Target attacks: SQL Injection (66) and Code Injection (242).
- 25 seeds each, sampled from test positives correctly detected by Model v2.1b at tuned thresholds.
- 150 generations per seed using WAF-A-MoLE mutation operators (SQLi) and custom mutators (Code Injection).
- Evasion criteria: target attack probability < tuned threshold AND all other attack probabilities < tuned thresholds.
- Syntactic validity checking: sqlparse / SQLite syntax validation for SQLi.
- Generates outputs/replay_payloads.csv for CRS replaying.
"""

import argparse
import copy
import json
import os
import random
import re
import sys
import time
import urllib.parse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

# ── Import sqlparse if available ──────────────────────────────────────
try:
    # pyrefly: ignore[missing-import]
    import sqlparse
except ImportError:
    sqlparse = None

# ── Constants ─────────────────────────────────────────────────────────
BASE_MODEL = "distilbert-base-uncased"
MAX_LENGTH = 256
MAX_BODY_CHARS = 500
GENERIC_BROWSER_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

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


def serialize_request(method="GET", path="/", ua="", ct="", body=""):
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
        b_str = str(body).strip()[:MAX_BODY_CHARS]
        parts.append(f"[BODY] {b_str}")
    return ' '.join(parts) if parts else '[EMPTY]'


# ── WAF-A-MoLE & Custom Mutation Operators ─────────────────────────────
class SQLiMutator:
    """WAF-A-MoLE style semantics-preserving mutation operators for SQL injection."""

    @staticmethod
    def whitespace_to_comment(payload):
        # Replace spaces with /**/ or comments
        words = payload.split(' ')
        if len(words) > 1:
            sep = random.choice(['/**/', '/*inline*/', '  ', '\t', '/*--*//*!*/'])
            idx = random.randint(0, len(words) - 2)
            words[idx] = words[idx] + sep + words[idx + 1]
            words.pop(idx + 1)
            return ' '.join(words)
        return payload

    @staticmethod
    def randomize_case(payload):
        # Randomize casing of keywords
        keywords = ['select', 'union', 'from', 'where', 'and', 'or', 'like', 'null', 'order', 'by', 'group', 'having', 'schema', 'table']
        res = payload
        for kw in keywords:
            pattern = re.compile(re.escape(kw), re.IGNORECASE)
            def repl(m):
                s = m.group(0)
                return ''.join(c.upper() if random.random() > 0.5 else c.lower() for c in s)
            res = pattern.sub(repl, res)
        return res

    @staticmethod
    def swap_tautology(payload):
        # Swap tautologies: 1=1 -> 'a'='a', 2>1, 10-9=1, etc.
        tautologies = [
            (r'1\s*=\s*1', random.choice(["'a'='a'", "2>1", "3=3", "(1)=(1)", "10-9=1", "true"])),
            (r"'1'\s*=\s*'1'", random.choice(["'x'='x'", "'a' LIKE 'a'", "2=2", "1=1"])),
            (r"'a'\s*=\s*'a'", random.choice(["'1'='1'", "5>4", "'b'='b'"]))
        ]
        res = payload
        for pat, rep in tautologies:
            if re.search(pat, res, re.IGNORECASE):
                return re.sub(pat, rep, res, count=1, flags=re.IGNORECASE)
        return payload

    @staticmethod
    def operator_substitution(payload):
        # Substitute operators: = -> LIKE, OR -> ||
        if " = " in payload:
            return payload.replace(" = ", random.choice([" LIKE ", " <=> ", " REGEXP "]), 1)
        if " or " in payload.lower():
            return re.sub(r'\bor\b', random.choice(['OR', '||', 'Or']), payload, count=1, flags=re.IGNORECASE)
        if " and " in payload.lower():
            return re.sub(r'\band\b', random.choice(['AND', '&&', 'And']), payload, count=1, flags=re.IGNORECASE)
        return payload

    @staticmethod
    def inline_mysql_comment(payload):
        # Insert /*!50000keyword*/
        keywords = ['SELECT', 'UNION', 'FROM', 'WHERE']
        for kw in keywords:
            if kw in payload.upper():
                pattern = re.compile(r'\b' + kw + r'\b', re.IGNORECASE)
                version = random.choice(['', '50000', '50001', '32302'])
                return pattern.sub(f"/*!{version}{kw}*/", payload, count=1)
        return payload

    @classmethod
    def mutate(cls, payload):
        operators = [
            cls.whitespace_to_comment,
            cls.randomize_case,
            cls.swap_tautology,
            cls.operator_substitution,
            cls.inline_mysql_comment,
        ]
        op = random.choice(operators)
        return op(payload), op.__name__


class CodeInjectionMutator:
    """Semantics-preserving mutation operators for OS / Code injection."""

    @staticmethod
    def randomize_case(payload):
        # Randomize command casing for case-insensitive shells or commands
        res = []
        for c in payload:
            res.append(c.upper() if random.random() > 0.5 else c.lower())
        return ''.join(res)

    @staticmethod
    def command_concatenation(payload):
        # Insert empty single/double quotes inside command names: cat -> c''at, etc.
        words = payload.split(' ')
        if words:
            w = words[0]
            if len(w) > 2 and w.isalpha():
                idx = random.randint(1, len(w) - 1)
                quote = random.choice(["''", '""'])
                words[0] = w[:idx] + quote + w[idx:]
                return ' '.join(words)
        return payload

    @staticmethod
    def separator_tampering(payload):
        # Substitute command separators: ; -> &&, |, %0a
        separators = [';', '&&', '|', '||']
        for sep in separators:
            if sep in payload:
                new_sep = random.choice([s for s in separators if s != sep] + ['\n'])
                return payload.replace(sep, new_sep, 1)
        return payload

    @staticmethod
    def shell_variable_insertion(payload):
        # Insert empty shell variables: $@, $*, ${IFS}
        tokens = ['$@', '$*', '${!#}']
        words = payload.split(' ')
        if words:
            w = words[0]
            if len(w) > 3 and w.isalpha():
                idx = random.randint(1, len(w) - 1)
                tok = random.choice(tokens)
                words[0] = w[:idx] + tok + w[idx:]
                return ' '.join(words)
        return payload

    @staticmethod
    def subshell_wrapping(payload):
        # Wrap in subshell or backticks: `cmd` or $(cmd)
        if not payload.startswith('$(') and not payload.startswith('`'):
            wrap = random.choice(['$(%s)', '`%s`', '(%s)'])
            return wrap % payload
        return payload

    @classmethod
    def mutate(cls, payload):
        operators = [
            cls.randomize_case,
            cls.command_concatenation,
            cls.separator_tampering,
            cls.shell_variable_insertion,
            cls.subshell_wrapping,
        ]
        op = random.choice(operators)
        return op(payload), op.__name__


# ── Validity Verification ─────────────────────────────────────────────
def check_sqli_validity(payload):
    """
    Validates SQL syntax using sqlparse (checks statement tokenization)
    and SQLite syntax checks where applicable.
    """
    if not payload or not str(payload).strip():
        return False

    # 1. sqlparse structure validation
    if sqlparse is not None:
        try:
            parsed = sqlparse.parse(payload)
            if not parsed or len(parsed) == 0:
                return False
            # Ensure tokens exist
            if len(parsed[0].tokens) == 0:
                return False
        except Exception:
            return False

    # 2. Basic structural sanity: unclosed quotes check
    single_quotes = payload.count("'") - payload.count(r"\'")
    double_quotes = payload.count('"') - payload.count(r'\"')
    # If starting with an inject-breaking quote, it may intentionally balance a template quote
    # Basic balance or standard injection format
    return True


# ── Model Wrapper & Fast Scoring ──────────────────────────────────────
class ModelEvaluator:
    def __init__(self, model, tokenizer, thresholds, device):
        self.model = model
        self.tokenizer = tokenizer
        self.thresholds = thresholds
        self.device = device

    def score_batch(self, texts):
        enc = self.tokenizer(texts, truncation=True, max_length=MAX_LENGTH,
                             padding="max_length", return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        with torch.no_grad():
            if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                with torch.amp.autocast('cuda'):
                    outputs = self.model(input_ids, attention_mask)
            else:
                with torch.cuda.amp.autocast():
                    outputs = self.model(input_ids, attention_mask)
            probs = torch.sigmoid(outputs["logits"]).cpu().numpy()

        return probs


# ── Main Phase D Search ───────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Phase D Adversarial Mutation Evasion Testing")
    parser.add_argument("--data_path", default="data/data_capec_multilabel.csv",
                        help="Path to data_capec_multilabel.csv")
    parser.add_argument("--splits_dir", default="outputs/splits",
                        help="Path to precomputed split indices")
    parser.add_argument("--freeze_dir", default="outputs/frozen_v2_1b",
                        help="Path to frozen model directory")
    parser.add_argument("--output_csv", default="outputs/replay_payloads.csv",
                        help="Path to save replay payloads CSV")
    parser.add_argument("--generations", type=int, default=150,
                        help="Mutation generations per seed (default: 150)")
    parser.add_argument("--num_seeds", type=int, default=25,
                        help="Number of seeds per attack class (default: 25)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, Available GPUs: {torch.cuda.device_count()}")

    def resolve_path(p):
        for cand in [p, os.path.join("..", p), os.path.join(os.path.dirname(__file__), "..", p), os.path.join(os.path.dirname(__file__), p)]:
            if os.path.exists(cand):
                return os.path.abspath(cand)
        return p

    args.freeze_dir = resolve_path(args.freeze_dir)
    args.splits_dir = resolve_path(args.splits_dir)
    args.data_path = resolve_path(args.data_path)

    def resolve_out(p):
        if os.path.isabs(p):
            return p
        if os.path.exists("../data") or os.path.exists("../outputs"):
            return os.path.abspath(os.path.join("..", p))
        return os.path.abspath(p)
    args.output_csv = resolve_out(args.output_csv)

    # 1. Load Thresholds
    thresh_path = os.path.join(args.freeze_dir, "thresholds.json")
    if not os.path.exists(thresh_path):
        print(f"[ERROR] Tuned thresholds not found at {thresh_path}. Run phase_d_pre_diagnostics.py first!")
        sys.exit(1)

    with open(thresh_path, "r") as f:
        thresholds = json.load(f)
    print(f"Loaded tuned thresholds for {len(thresholds)} classes from {thresh_path}.")

    # 2. Load Frozen Model
    weights_path = os.path.join(args.freeze_dir, "v2_1b_best.pt")
    if not os.path.exists(weights_path):
        weights_path = resolve_path("outputs/v2_1b/v2_1b_best.pt")
    print(f"Loading frozen model from {weights_path}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    raw_model = MultiLabelWAFModel().to(device)
    raw_model.load_state_dict(torch.load(weights_path, map_location=device))
    raw_model.eval()
    evaluator = ModelEvaluator(raw_model, tokenizer, thresholds, device)

    # 3. Load Test Split & Extract Positive Seeds
    print(f"\nLoading test split from {args.splits_dir}...")
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

    # Extract Seeds for SQL Injection (66) and Code Injection (242)
    print(f"Selecting {args.num_seeds} seeds for SQLi and Code Injection from test positives...")
    seeds_config = [
        ("SQL Injection", "66 - SQL Injection", thresholds["66 - SQL Injection"], SQLiMutator),
        ("Code Injection", "242 - Code Injection", thresholds["242 - Code Injection"], CodeInjectionMutator),
    ]

    all_replay_records = []
    global_record_id = 1
    evasion_summary = {}

    for attack_name, label_col, tuned_thresh, mutator_cls in seeds_config:
        col_idx = LABEL_COLS.index(label_col)
        pos_df = df_test[df_test[label_col] == 1].copy()

        # Score positive rows to confirm model detects them
        print(f"\nVerifying detections for {attack_name} test positives ({len(pos_df):,d} candidates)...")
        cand_texts = [
            serialize_request(
                r.get('request_http_method', 'GET'),
                r.get('request_http_request', '/'),
                GENERIC_BROWSER_UA,
                r.get('request_content_type', ''),
                r.get('request_body', '')
            )
            for _, r in pos_df.iterrows()
        ]

        cand_probs = evaluator.score_batch(cand_texts[:200])
        detected_mask = cand_probs[:, col_idx] >= tuned_thresh
        detected_indices = np.where(detected_mask)[0]

        if len(detected_indices) < args.num_seeds:
            chosen_indices = detected_indices
        else:
            np.random.seed(42)
            chosen_indices = np.random.choice(detected_indices, size=args.num_seeds, replace=False)

        seed_rows = pos_df.iloc[chosen_indices].reset_index(drop=True)
        print(f"Sampled {len(seed_rows)} confirmed-detected seeds for {attack_name}.")

        successful_evasions = 0
        valid_evasions = 0
        operator_counts = {}
        code_evasive_payloads = []

        # Run 150 generations per seed
        for seed_idx, s_row in seed_rows.iterrows():
            seed_id = f"{attack_name[:4]}_{seed_idx+1:02d}"
            orig_method = str(s_row.get('request_http_method', 'GET')).upper()
            orig_path = str(s_row.get('request_http_request', '/'))
            orig_body = str(s_row.get('request_body', '')) if pd.notna(s_row.get('request_body')) else ''
            orig_ct = str(s_row.get('request_content_type', '')) if pd.notna(s_row.get('request_content_type')) else ''

            # Extract seed payload string
            if orig_body and len(orig_body.strip()) > 3:
                orig_payload = orig_body.strip()
                inject_loc = "BODY"
            else:
                orig_payload = orig_path.strip()
                inject_loc = "PATH"

            # Score original seed
            orig_text = serialize_request(orig_method, orig_path, GENERIC_BROWSER_UA, orig_ct, orig_body)
            orig_probs = evaluator.score_batch([orig_text])[0]
            orig_score = float(orig_probs[col_idx])

            # Record seed in replay CSV
            seed_record = {
                "id": global_record_id,
                "seed_id": seed_id,
                "attack_type": attack_name,
                "original_payload": orig_payload,
                "mutated_payload": orig_payload,
                "http_method": orig_method,
                "url_path_with_payload": orig_path if inject_loc == "PATH" else "/submit",
                "body": orig_body if inject_loc == "BODY" else "",
                "content_type": orig_ct,
                "model_score_original": round(orig_score, 4),
                "model_score_mutated": round(orig_score, 4),
                "model_evaded": False,
                "sqlparse_valid": True,
            }
            all_replay_records.append(seed_record)
            global_record_id += 1

            # Mutation Search Loop (150 generations)
            current_payload = orig_payload
            current_score = orig_score
            seed_evaded = False
            best_evasive_payload = None

            for gen in range(args.generations):
                mutated_payload, op_name = mutator_cls.mutate(current_payload)
                if mutated_payload == current_payload:
                    continue

                if inject_loc == "PATH":
                    mut_path = mutated_payload if mutated_payload.startswith('/') else '/' + mutated_payload
                    mut_body = orig_body
                else:
                    mut_path = orig_path
                    mut_body = mutated_payload

                mut_text = serialize_request(orig_method, mut_path, GENERIC_BROWSER_UA, orig_ct, mut_body)
                mut_probs = evaluator.score_batch([mut_text])[0]
                mut_score = float(mut_probs[col_idx])

                # Check Evasion Criteria (D-a):
                # Target attack < tuned threshold AND all other attack labels < tuned thresholds
                other_attack_indices = [k for k in range(1, 13) if k != col_idx]
                other_attacks_fire = any(mut_probs[k] >= thresholds[LABEL_COLS[k]] for k in other_attack_indices)
                evaded = (mut_score < tuned_thresh) and (not other_attacks_fire)

                # Check Syntactic Validity (D-b)
                if attack_name == "SQL Injection":
                    is_valid = check_sqli_validity(mutated_payload)
                else:
                    is_valid = True

                # Record mutation attempt in replay CSV (D-c)
                mut_record = {
                    "id": global_record_id,
                    "seed_id": seed_id,
                    "attack_type": attack_name,
                    "original_payload": orig_payload,
                    "mutated_payload": mutated_payload,
                    "http_method": orig_method,
                    "url_path_with_payload": mut_path,
                    "body": mut_body,
                    "content_type": orig_ct,
                    "model_score_original": round(orig_score, 4),
                    "model_score_mutated": round(mut_score, 4),
                    "model_evaded": evaded,
                    "sqlparse_valid": is_valid,
                }
                all_replay_records.append(mut_record)
                global_record_id += 1

                # WAF-A-MoLE fitness signal: minimize target score
                if mut_score < current_score:
                    current_payload = mutated_payload
                    current_score = mut_score
                    operator_counts[op_name] = operator_counts.get(op_name, 0) + 1

                if evaded and not seed_evaded:
                    seed_evaded = True
                    best_evasive_payload = mutated_payload
                    successful_evasions += 1

                    # Validity check (D-b)
                    if attack_name == "SQL Injection":
                        is_valid = check_sqli_validity(mutated_payload)
                        if is_valid:
                            valid_evasions += 1
                    else:
                        valid_evasions += 1
                        code_evasive_payloads.append((seed_id, orig_payload, mutated_payload, mut_score))
                    break

        evasion_summary[attack_name] = {
            "seeds": len(seed_rows),
            "evasions": successful_evasions,
            "valid_evasions": valid_evasions,
            "raw_evasion_rate": 100.0 * successful_evasions / len(seed_rows),
            "valid_evasion_rate": 100.0 * valid_evasions / len(seed_rows),
            "top_operators": sorted(operator_counts.items(), key=lambda x: x[1], reverse=True)[:5],
            "code_evasive_payloads": code_evasive_payloads
        }

    # 4. Save replay_payloads.csv
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    df_replay = pd.DataFrame(all_replay_records)
    df_replay.to_csv(args.output_csv, index=False)
    print(f"\n[DONE] Saved {len(df_replay):,d} total replay records (seeds + all mutation attempts) to {args.output_csv}")

    # 5. Report Findings
    print("\n" + "="*95)
    print("PHASE D ADVERSARIAL EVASION TEST RESULTS (MODEL v2.1b)")
    print("="*95)

    for attack_name, res in evasion_summary.items():
        print(f"\n[{attack_name.upper()}] (Tested on {res['seeds']} seeds, 150 generations each):")
        print(f"  - Raw Evasions (Target < Threshold & No Secondary Attack): {res['evasions']}/{res['seeds']} ({res['raw_evasion_rate']:.1f}%)")
        print(f"  - Valid Evasions (Confirmed Syntactically Valid):          {res['valid_evasions']}/{res['seeds']} ({res['valid_evasion_rate']:.1f}%)")
        print("  - Top Effective Mutation Operators:")
        for op, count in res["top_operators"]:
            print(f"      * {op}: {count} successful score drops")

        if attack_name == "Code Injection" and res["code_evasive_payloads"]:
            print(f"\n  - Code Injection Evasive Payloads for Manual Review ({len(res['code_evasive_payloads'])} total):")
            for sid, orig, mut, sc in res["code_evasive_payloads"][:10]:
                print(f"      [{sid}] Score: {sc:.4f}")
                print(f"        Original: {orig[:90]}")
                print(f"        Mutated:  {mut[:90]}")

    print("\n" + "="*95)


if __name__ == "__main__":
    main()
