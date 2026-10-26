"""
Configuration for 4thwall_v3 ML WAF (Frozen Model v2.1b).
"""

MODEL_NAME = "distilbert-base-uncased"
HF_MODEL_REPO = "viitheone/4thwall-waf-v2.1b"
LOCAL_MODEL_PATH = "models/huggingface_model"
FROZEN_MODEL_PATH = "models/v2_1b"

# Maximum sequence length (tokens) for DistilBERT
MAX_LENGTH = 256

# Maximum body characters serialized to prevent token starvation of URI components
MAX_BODY_CHARS = 500

# Canonical browser fallback User-Agent (prevents false positives on scanner heuristics)
DEFAULT_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

# 13 Multi-Label CAPEC Classification Categories
LABEL_COLS = [
    '000 - Normal',                           # Index 0  (Benign baseline)
    '272 - Protocol Manipulation',            # Index 1  (ALERT tier)
    '242 - Code Injection',                   # Index 2  (BLOCK tier)
    '88 - OS Command Injection',              # Index 3  (BLOCK tier, merged CAPEC-248)
    '126 - Path Traversal',                   # Index 4  (BLOCK tier)
    '66 - SQL Injection',                     # Index 5  (BLOCK tier)
    '16 - Dictionary-based Password Attack',  # Index 6  (BLOCK tier)
    '310 - Scanning for Vulnerable Software', # Index 7  (ALERT tier)
    '153 - Input Data Manipulation',          # Index 8  (BLOCK tier)
    '274 - HTTP Verb Tampering',              # Index 9  (BLOCK tier)
    '194 - Fake the Source of Data',          # Index 10 (BLOCK tier)
    '34 - HTTP Response Splitting',           # Index 11 (BLOCK tier)
    '33 - HTTP Request Smuggling'             # Index 12 (BLOCK tier)
]

# Two-Tier Enforcement Policy: 10 automated BLOCK vectors
BLOCK_CLASSES = {
    '242 - Code Injection',
    '88 - OS Command Injection',
    '126 - Path Traversal',
    '66 - SQL Injection',
    '16 - Dictionary-based Password Attack',
    '153 - Input Data Manipulation',
    '274 - HTTP Verb Tampering',
    '194 - Fake the Source of Data',
    '34 - HTTP Response Splitting',
    '33 - HTTP Request Smuggling'
}

# Two-Tier Enforcement Policy: 2 passive ALERT vectors
ALERT_CLASSES = {
    '272 - Protocol Manipulation',
    '310 - Scanning for Vulnerable Software'
}

# Validation-calibrated decision thresholds (DO NOT USE default 0.50)
CALIBRATED_THRESHOLDS = {
    "000 - Normal": 0.55,
    "272 - Protocol Manipulation": 0.99,
    "242 - Code Injection": 0.65,
    "88 - OS Command Injection": 0.90,
    "126 - Path Traversal": 0.90,
    "66 - SQL Injection": 0.70,
    "16 - Dictionary-based Password Attack": 0.60,
    "310 - Scanning for Vulnerable Software": 0.55,
    "153 - Input Data Manipulation": 0.65,
    "274 - HTTP Verb Tampering": 0.90,
    "194 - Fake the Source of Data": 0.90,
    "34 - HTTP Response Splitting": 0.60,
    "33 - HTTP Request Smuggling": 0.95
}

# Training / legacy parameters preserved for backward compatibility
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
TRAIN_SPLIT = 0.8
ML_THRESHOLD_BLOCK = 0.9
ML_THRESHOLD_ALERT = 0.6
MODEL_SAVE_PATH = "./models/waf_model"
