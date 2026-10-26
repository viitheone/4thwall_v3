"""
Phase E1 Item (a): WAF-A-MoLE Audit & SQLi Logit Search
1. Tokenize every seed and mutant with distilbert-base-uncased.
   Report per mutation operator the % of mutants whose input_ids are identical to the seed's.
2. Re-run the SQLi search (150 generations, 25 seeds) using the target-label LOGIT (pre-sigmoid) as fitness.
   Report mutants evaluated, unique after tokenization, and per-seed min logit/score reached.
   Keep old CSVs; write results to outputs/phase_e1_sqli_logit_search.csv.
"""

import json
import os
import random
import re
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

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

# SQLi Mutation Operators
class SQLiMutator:
    @staticmethod
    def whitespace_to_comment(payload):
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
        if " = " in payload:
            return payload.replace(" = ", random.choice([" LIKE ", " <=> ", " REGEXP "]), 1)
        if " or " in payload.lower():
            return re.sub(r'\bor\b', random.choice(['OR', '||', 'Or']), payload, count=1, flags=re.IGNORECASE)
        if " and " in payload.lower():
            return re.sub(r'\band\b', random.choice(['AND', '&&', 'And']), payload, count=1, flags=re.IGNORECASE)
        return payload

    @staticmethod
    def inline_mysql_comment(payload):
        keywords = ['SELECT', 'UNION', 'FROM', 'WHERE']
        for kw in keywords:
            if kw in payload.upper():
                pattern = re.compile(r'\b' + kw + r'\b', re.IGNORECASE)
                version = random.choice(['', '50000', '50001', '32302'])
                return pattern.sub(f"/*!{version}{kw}*/", payload, count=1)
        return payload

    @classmethod
    def get_operators(cls):
        return [
            ("whitespace_to_comment", cls.whitespace_to_comment),
            ("randomize_case", cls.randomize_case),
            ("swap_tautology", cls.swap_tautology),
            ("operator_substitution", cls.operator_substitution),
            ("inline_mysql_comment", cls.inline_mysql_comment),
        ]

    @classmethod
    def mutate(cls, payload):
        ops = cls.get_operators()
        name, fn = random.choice(ops)
        return fn(payload), name

# Code Injection Mutation Operators
class CodeInjectionMutator:
    @staticmethod
    def randomize_case(payload):
        return ''.join(c.upper() if random.random() > 0.5 else c.lower() for c in payload)

    @staticmethod
    def command_concatenation(payload):
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
        separators = [';', '&&', '|', '||']
        for sep in separators:
            if sep in payload:
                new_sep = random.choice([s for s in separators if s != sep] + ['\n'])
                return payload.replace(sep, new_sep, 1)
        return payload

    @staticmethod
    def shell_variable_insertion(payload):
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
        if not payload.startswith('$(') and not payload.startswith('`'):
            wrap = random.choice(['$(%s)', '`%s`', '(%s)'])
            return wrap % payload
        return payload

    @classmethod
    def get_operators(cls):
        return [
            ("randomize_case", cls.randomize_case),
            ("command_concatenation", cls.command_concatenation),
            ("separator_tampering", cls.separator_tampering),
            ("shell_variable_insertion", cls.shell_variable_insertion),
            ("subshell_wrapping", cls.subshell_wrapping),
        ]

def main():
    print("=" * 100)
    print("PHASE E1 (a): WAF-A-MoLE AUDIT & LOGIT-BASED SEARCH")
    print("=" * 100)

    device = torch.device("cpu")
    print(f"Device: {device}")

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # 1. Audit Existing Replay Payloads
    replay_csv = "outputs/replay_payloads.csv"
    if not os.path.exists(replay_csv):
        replay_csv = "model_v2.0/outputs/replay_payloads.csv"
    
    df_replay = pd.read_csv(replay_csv)
    print(f"Loaded {len(df_replay):,d} records from {replay_csv}")

    # Extract seeds
    seed_rows = df_replay[df_replay['original_payload'] == df_replay['mutated_payload']].copy()
    sqli_seeds = seed_rows[seed_rows['attack_type'] == 'SQL Injection'].reset_index(drop=True)
    code_seeds = seed_rows[seed_rows['attack_type'] == 'Code Injection'].reset_index(drop=True)

    print(f"Found {len(sqli_seeds)} SQLi seeds and {len(code_seeds)} Code Injection seeds.")

    # A) Tokenization collision audit on existing CSV
    print("\n--- [PART 1]: Tokenization Invariance on Existing Replay Payloads ---")
    mutants = df_replay[df_replay['original_payload'] != df_replay['mutated_payload']].copy()
    
    identical_count = 0
    total_mutants = len(mutants)
    for _, row in tqdm(mutants.iterrows(), total=total_mutants, desc="Tokenizing existing mutants"):
        orig_text = serialize_request(row['http_method'], row['url_path_with_payload'], GENERIC_BROWSER_UA, row['content_type'], row['original_payload'])
        mut_text = serialize_request(row['http_method'], row['url_path_with_payload'], GENERIC_BROWSER_UA, row['content_type'], row['mutated_payload'])
        
        ids_orig = tokenizer.encode(orig_text, truncation=True, max_length=MAX_LENGTH)
        ids_mut = tokenizer.encode(mut_text, truncation=True, max_length=MAX_LENGTH)
        if ids_orig == ids_mut:
            identical_count += 1

    pct_identical = 100.0 * identical_count / total_mutants if total_mutants > 0 else 0
    print(f"Overall Existing Mutants: {identical_count:,d} / {total_mutants:,d} ({pct_identical:.2f}%) have EXACTLY IDENTICAL input_ids to their seed!")

    # B) Per-Operator Token Collision Rate
    print("\n--- [PART 2]: Per-Operator Token Collision Rate Audit ---")
    print("Testing 250 trial mutations per operator across seeds...\n")
    print(f"{'Attack Class':<18} {'Mutation Operator':<28} {'Trials':>8} {'Identical IDs':>14} {'Collision %':>12}")
    print("-" * 85)

    random.seed(42)
    operator_results = []

    # SQLi operators
    for op_name, op_fn in SQLiMutator.get_operators():
        ident = 0
        trials = 250
        for _ in range(trials):
            s_row = sqli_seeds.sample(1, random_state=random.randint(0, 100000)).iloc[0]
            payload = str(s_row['original_payload'])
            mut = op_fn(payload)
            orig_text = serialize_request("GET", f"/?id={payload}", GENERIC_BROWSER_UA, "", "")
            mut_text = serialize_request("GET", f"/?id={mut}", GENERIC_BROWSER_UA, "", "")
            if tokenizer.encode(orig_text, truncation=True, max_length=MAX_LENGTH) == tokenizer.encode(mut_text, truncation=True, max_length=MAX_LENGTH):
                ident += 1
        coll_pct = 100.0 * ident / trials
        print(f"{'SQL Injection':<18} {op_name:<28} {trials:>8} {ident:>14} {coll_pct:>11.2f}%")
        operator_results.append(("SQL Injection", op_name, trials, ident, coll_pct))

    # Code Injection operators
    for op_name, op_fn in CodeInjectionMutator.get_operators():
        ident = 0
        trials = 250
        for _ in range(trials):
            s_row = code_seeds.sample(1, random_state=random.randint(0, 100000)).iloc[0]
            payload = str(s_row['original_payload'])
            mut = op_fn(payload)
            orig_text = serialize_request("GET", f"/?id={payload}", GENERIC_BROWSER_UA, "", "")
            mut_text = serialize_request("GET", f"/?id={mut}", GENERIC_BROWSER_UA, "", "")
            if tokenizer.encode(orig_text, truncation=True, max_length=MAX_LENGTH) == tokenizer.encode(mut_text, truncation=True, max_length=MAX_LENGTH):
                ident += 1
        coll_pct = 100.0 * ident / trials
        print(f"{'Code Injection':<18} {op_name:<28} {trials:>8} {ident:>14} {coll_pct:>11.2f}%")
        operator_results.append(("Code Injection", op_name, trials, ident, coll_pct))

    print("-" * 85)

    # C) Re-run SQLi Search (150 generations) using Logit Fitness
    print("\n--- [PART 3]: SQLi Search Re-Run with Target LOGIT Fitness (150 gens) ---")

    # Load frozen weights
    weights_path = "model_v2.0/outputs/v2_1b/v2_1b_best.pt"
    if not os.path.exists(weights_path):
        weights_path = "outputs/frozen_v2_1b/v2_1b_best.pt"
    if not os.path.exists(weights_path):
        weights_path = "outputs/v2_1b/v2_1b_best.pt"

    print(f"Loading weights from {weights_path}...")
    model = MultiLabelWAFModel().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # Load thresholds
    thresh_path = "outputs/frozen_v2_1b/thresholds.json"
    with open(thresh_path, "r") as f:
        thresholds = json.load(f)
    sqli_thresh = thresholds["66 - SQL Injection"]
    col_idx = LABEL_COLS.index("66 - SQL Injection")
    print(f"Frozen SQLi threshold: {sqli_thresh} (col {col_idx})")

    def evaluate_text(text):
        enc = tokenizer([text], truncation=True, max_length=MAX_LENGTH, padding="max_length", return_tensors="pt")
        with torch.no_grad():
            out = model(enc["input_ids"], enc["attention_mask"])
            logits = out["logits"][0]
            probs = torch.sigmoid(logits)
        input_ids_tuple = tuple(enc["input_ids"][0].tolist())
        return float(logits[col_idx]), float(probs[col_idx]), input_ids_tuple

    # Run 150 generations per SQLi seed
    random.seed(42)
    np.random.seed(42)

    search_records = []
    per_seed_summary = []
    all_evaluated_token_hashes = set()

    total_mutants_evaluated = 0
    n_generations = 150

    print(f"Running search on {len(sqli_seeds)} seeds x {n_generations} generations...")

    for s_idx, s_row in tqdm(sqli_seeds.iterrows(), total=len(sqli_seeds), desc="SQLi Seeds Search"):
        sid = s_row['seed_id']
        orig_payload = str(s_row['original_payload'])
        orig_method = str(s_row.get('http_method', 'GET'))
        orig_path = str(s_row.get('url_path_with_payload', '/'))
        orig_ct = str(s_row.get('content_type', '')) if pd.notna(s_row.get('content_type')) else ''
        orig_body = str(s_row.get('body', '')) if pd.notna(s_row.get('body')) else ''

        # Evaluate seed
        seed_text = serialize_request(orig_method, orig_path, GENERIC_BROWSER_UA, orig_ct, orig_body)
        init_logit, init_score, seed_ids = evaluate_text(seed_text)
        all_evaluated_token_hashes.add(seed_ids)

        current_payload = orig_payload
        current_logit = init_logit
        current_score = init_score

        min_logit_reached = init_logit
        min_score_reached = init_score

        seed_unique_token_sets = {seed_ids}
        seed_eval_count = 0

        # Initial seed record
        search_records.append({
            "seed_id": sid,
            "generation": 0,
            "operator": "SEED",
            "payload": orig_payload,
            "target_logit": init_logit,
            "target_score": init_score,
            "evaded": init_score < sqli_thresh,
            "unique_tokens": True,
        })

        for gen in range(1, n_generations + 1):
            mutated, op_name = SQLiMutator.mutate(current_payload)
            if mutated == current_payload:
                continue

            # Determine where payload is injected
            if orig_body and len(orig_body.strip()) > 3:
                mut_path = orig_path
                mut_body = mutated
            else:
                mut_path = mutated if mutated.startswith('/') else '/' + mutated
                mut_body = orig_body

            mut_text = serialize_request(orig_method, mut_path, GENERIC_BROWSER_UA, orig_ct, mut_body)
            mut_logit, mut_score, mut_ids = evaluate_text(mut_text)

            total_mutants_evaluated += 1
            seed_eval_count += 1

            is_new_tokens = mut_ids not in all_evaluated_token_hashes
            all_evaluated_token_hashes.add(mut_ids)
            seed_unique_token_sets.add(mut_ids)

            # Record
            search_records.append({
                "seed_id": sid,
                "generation": gen,
                "operator": op_name,
                "payload": mutated,
                "target_logit": mut_logit,
                "target_score": mut_score,
                "evaded": mut_score < sqli_thresh,
                "unique_tokens": is_new_tokens,
            })

            # LOGIT FITNESS: minimize logit
            if mut_logit < current_logit:
                current_payload = mutated
                current_logit = mut_logit
                current_score = mut_score

            if mut_logit < min_logit_reached:
                min_logit_reached = mut_logit
            if mut_score < min_score_reached:
                min_score_reached = mut_score

        per_seed_summary.append({
            "seed_id": sid,
            "initial_logit": round(init_logit, 4),
            "min_logit": round(min_logit_reached, 4),
            "logit_drop": round(init_logit - min_logit_reached, 4),
            "initial_score": round(init_score, 4),
            "min_score": round(min_score_reached, 4),
            "score_drop": round(init_score - min_score_reached, 4),
            "thresh": sqli_thresh,
            "evaded": min_score_reached < sqli_thresh,
            "evaluated": seed_eval_count,
            "unique_token_patterns": len(seed_unique_token_sets),
        })

    # Save search records to CSV
    os.makedirs("outputs", exist_ok=True)
    df_search_out = pd.DataFrame(search_records)
    out_csv = "outputs/phase_e1_sqli_logit_search.csv"
    df_search_out.to_csv(out_csv, index=False)
    print(f"\n[DONE] Saved {len(df_search_out):,d} search records to {out_csv}")

    # Summary table
    df_summary = pd.DataFrame(per_seed_summary)
    print("\n" + "=" * 115)
    print("PER-SEED LOGIT-BASED SEARCH RESULTS (SQL INJECTION, 150 GENERATIONS)")
    print("=" * 115)
    print(df_summary.to_string(index=False))
    print("=" * 115)

    print("\n--- GRAND SEARCH TOTALS ---")
    print(f"Total Mutants Evaluated:            {total_mutants_evaluated:,d}")
    print(f"Total Unique Token Sequences Found: {len(all_evaluated_token_hashes):,d}")
    print(f"Token Sequence Novelty Rate:        {100.0 * len(all_evaluated_token_hashes) / (total_mutants_evaluated + len(sqli_seeds)):.2f}%")
    print(f"Mean Initial Logit:                 {df_summary['initial_logit'].mean():.4f}")
    print(f"Mean Min Logit Reached:             {df_summary['min_logit'].mean():.4f}")
    print(f"Mean Logit Drop:                    {df_summary['logit_drop'].mean():.4f}")
    print(f"Mean Initial Score:                 {df_summary['initial_score'].mean():.4f}")
    print(f"Mean Min Score Reached:             {df_summary['min_score'].mean():.4f}")
    print(f"Max Score Drop:                     {df_summary['score_drop'].max():.4f}")
    print(f"Seeds Evaded (score < 0.70):        {df_summary['evaded'].sum()} / {len(df_summary)}")
    print("=" * 115)

if __name__ == "__main__":
    main()
