"""
Model v2.0 — Configuration
All hyperparameters, label definitions, and paths in one place.
Fresh DistilBERT base — no v1 weights reused.
"""

import os

# ─── Base model ───────────────────────────────────────────────
BASE_MODEL = "distilbert-base-uncased"  # Fresh pretrained only
MAX_LENGTH = 256  # Includes method + headers + body now

# ─── Training hyperparameters ─────────────────────────────────
BATCH_SIZE = 64   # Fills T4 GPU properly (was 16 → only 23% GPU usage)
LEARNING_RATE = 2e-5
NUM_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 3        # Stop if val macro-F1 doesn't improve
DROPOUT = 0.3
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1                 # Fraction of total steps for LR warmup

# ─── Data split ───────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# Use time-based split instead of random shuffle to prevent
# leakage from same-attacker sessions appearing in both train and test.
# Set to False to use random split (not recommended for honeypot data).
TIME_BASED_SPLIT = True
TIMESTAMP_COLUMN = "timestamp"
# Apache Common Log Format: '17/Jul/2020:12:23:34 +0100'
# pd.to_datetime can't auto-detect this — must specify explicitly.
TIMESTAMP_FORMAT = "%d/%b/%Y:%H:%M:%S %z"

# Session key for group-based splitting (fallback if time split fails).
# Only 1 unique src_ip in SR-BH 2020 (honeypot IP), so UA is the session key.
SESSION_KEY_COLUMN = "request_user_agent"

# ─── Label merging ────────────────────────────────────────────
# CAPEC-248 (Command Injection) had exactly 1 sample — impossible to learn.
# CAPEC-248 is a child of CAPEC-88 (OS Command Injection) in the CAPEC hierarchy.
# We merge 248 into 88 before training. This is standard practice:
# "insufficient samples for CAPEC-248, merged with parent category CAPEC-88"
LABEL_MERGE_MAP = {
    "248 - Command Injection": "88 - OS Command Injection",
}

# ─── CAPEC label definitions (after merge) ────────────────────
# 13 classes: 1 normal + 12 attack categories
# "248 - Command Injection" merged into "88 - OS Command Injection"
CAPEC_LABELS = [
    "000 - Normal",
    "272 - Protocol Manipulation",
    "242 - Code Injection",
    "88 - OS Command Injection",   # includes merged CAPEC-248
    "126 - Path Traversal",
    "66 - SQL Injection",
    "16 - Dictionary-based Password Attack",
    "310 - Scanning for Vulnerable Software",
    "153 - Input Data Manipulation",
    "274 - HTTP Verb Tampering",
    "194 - Fake the Source of Data",
    "34 - HTTP Response Splitting",
    "33 - HTTP Request Smuggling",
]

NUM_LABELS = len(CAPEC_LABELS)  # 13

# Human-readable label descriptions
CAPEC_DESCRIPTIONS = {
    "000 - Normal":                          "Legitimate traffic",
    "272 - Protocol Manipulation":           "Protocol Manipulation",
    "242 - Code Injection":                  "Code Injection",
    "88 - OS Command Injection":             "OS / Command Injection (incl. CAPEC-248)",
    "126 - Path Traversal":                  "Path Traversal",
    "66 - SQL Injection":                    "SQL Injection",
    "16 - Dictionary-based Password Attack": "Brute-force / Dictionary Password Attack",
    "310 - Scanning for Vulnerable Software":"Vulnerability Scanning",
    "153 - Input Data Manipulation":         "Input Data Manipulation",
    "274 - HTTP Verb Tampering":             "HTTP Verb Tampering",
    "194 - Fake the Source of Data":         "Data Source Forgery",
    "34 - HTTP Response Splitting":          "HTTP Response Splitting",
    "33 - HTTP Request Smuggling":           "HTTP Request Smuggling",
}

# ─── Serialization settings ───────────────────────────────────
TEXT_COLUMN = "request_http_request"

# Max characters for request_body before truncation.
# Large POST bodies (file uploads, etc.) can blow past DistilBERT's 512-token limit.
# This is a reproducibility detail — we log how often truncation occurs.
MAX_BODY_CHARS = 500

# Cookie field: excluded by default because raw cookie values are usually
# session tokens (random noise) that don't predict CAPEC category.
# Enable for ablation study to verify whether cookies actually help.
INCLUDE_COOKIE = False

# ─── Inference thresholds ─────────────────────────────────────
SIGMOID_THRESHOLD = 0.5  # Per-label classification threshold

# ─── Paths (relative to model_v2.0/) ─────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
