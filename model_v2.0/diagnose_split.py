"""
Diagnostic: Inspect timestamps + measure session leakage in current stratified split.
"""
import pandas as pd
import numpy as np
from dataset import load_dataset, stratified_multilabel_split
from config import TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED

# ── Step 1: Inspect raw timestamps ──────────────────────────────
print("=" * 60)
print("  STEP 1: Timestamp Inspection")
print("=" * 60)

df = pd.read_csv("data/data_capec_multilabel.csv", low_memory=False)
ts_col = "timestamp"

print(f"\n  Column dtype: {df[ts_col].dtype}")
print(f"  5 sample raw values:")
for i, val in enumerate(df[ts_col].dropna().head(5)):
    print(f"    [{i}] {repr(val)}")

# Try different parse strategies
print(f"\n  Parse attempts:")

# 1. Default pd.to_datetime
ts1 = pd.to_datetime(df[ts_col], errors='coerce')
nat1 = ts1.isna().sum()
print(f"    Default pd.to_datetime:  {nat1}/{len(df)} NaT ({100*nat1/len(df):.1f}%)")

# 2. Unix epoch (seconds)
try:
    ts2 = pd.to_datetime(df[ts_col], unit='s', errors='coerce')
    nat2 = ts2.isna().sum()
    print(f"    Unix epoch (seconds):    {nat2}/{len(df)} NaT ({100*nat2/len(df):.1f}%)")
    if nat2 < nat1:
        print(f"      Sample parsed: {ts2.dropna().head(3).tolist()}")
except:
    print(f"    Unix epoch (seconds):    FAILED")

# 3. Unix epoch (milliseconds)
try:
    ts3 = pd.to_datetime(df[ts_col], unit='ms', errors='coerce')
    nat3 = ts3.isna().sum()
    print(f"    Unix epoch (ms):         {nat3}/{len(df)} NaT ({100*nat3/len(df):.1f}%)")
    if nat3 < nat1:
        print(f"      Sample parsed: {ts3.dropna().head(3).tolist()}")
except:
    print(f"    Unix epoch (ms):         FAILED")

# 4. Try common string formats
for fmt_name, fmt in [
    ("DD/MM/YYYY HH:MM:SS", "%d/%m/%Y %H:%M:%S"),
    ("MM/DD/YYYY HH:MM:SS", "%m/%d/%Y %H:%M:%S"),
    ("YYYY-MM-DD HH:MM:SS", "%Y-%m-%d %H:%M:%S"),
    ("YYYY/MM/DD HH:MM:SS", "%Y/%m/%d %H:%M:%S"),
]:
    try:
        ts_try = pd.to_datetime(df[ts_col], format=fmt, errors='coerce')
        nat_try = ts_try.isna().sum()
        pct = 100 * nat_try / len(df)
        status = "OK" if pct < 10 else "FAIL"
        print(f"    {fmt_name:25s} {nat_try}/{len(df)} NaT ({pct:.1f}%) [{status}]")
        if nat_try < len(df) * 0.1:
            print(f"      Sample parsed: {ts_try.dropna().head(3).tolist()}")
    except:
        print(f"    {fmt_name:25s} FAILED")

# Check if numeric
print(f"\n  Numeric check:")
numeric_vals = pd.to_numeric(df[ts_col], errors='coerce')
num_valid = numeric_vals.notna().sum()
print(f"    Numeric values: {num_valid}/{len(df)} ({100*num_valid/len(df):.1f}%)")
if num_valid > 0:
    print(f"    Range: {numeric_vals.min()} to {numeric_vals.max()}")
    print(f"    Sample: {numeric_vals.dropna().head(5).tolist()}")


# ── Step 2: Measure leakage in current stratified split ──────────
print(f"\n{'=' * 60}")
print("  STEP 2: Session Leakage in Current Stratified Split")
print("=" * 60)

texts, labels, label_columns, df_loaded = load_dataset("data/data_capec_multilabel.csv")

# Run the stratified split
(train_texts, train_labels,
 val_texts, val_labels,
 test_texts, test_labels) = stratified_multilabel_split(
    texts, labels, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED,
)

# Get indices for each split by matching back to df
# Since stratified_multilabel_split returns shuffled data, we need the indices
# Re-run to get actual indices
texts_arr = np.array(texts, dtype=object)
labels_arr = np.array(labels)
label_sums = labels_arr.sum(axis=0)
sorted_by_freq = np.argsort(label_sums)
strat_cols = [i for i in sorted_by_freq if label_sums[i] > 0][:3]
strat_key = ["".join(str(int(labels_arr[i, c])) for c in strat_cols) for i in range(len(texts))]

from sklearn.model_selection import train_test_split
val_test_ratio = VAL_RATIO + TEST_RATIO
try:
    train_idx, valtest_idx = train_test_split(
        np.arange(len(texts)), test_size=val_test_ratio,
        random_state=RANDOM_SEED, stratify=strat_key,
    )
except:
    train_idx, valtest_idx = train_test_split(
        np.arange(len(texts)), test_size=val_test_ratio, random_state=RANDOM_SEED,
    )

relative_test = TEST_RATIO / val_test_ratio
try:
    valtest_strat = [strat_key[i] for i in valtest_idx]
    val_idx, test_idx = train_test_split(
        valtest_idx, test_size=relative_test,
        random_state=RANDOM_SEED, stratify=valtest_strat,
    )
except:
    val_idx, test_idx = train_test_split(
        valtest_idx, test_size=relative_test, random_state=RANDOM_SEED,
    )

print(f"\n  Split sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

# Check User-Agent overlap
ua_col = 'request_user_agent'
if ua_col in df_loaded.columns:
    train_ua = set(df_loaded.iloc[train_idx][ua_col].dropna().unique())
    test_ua = set(df_loaded.iloc[test_idx][ua_col].dropna().unique())
    overlap_ua = train_ua & test_ua
    total_unique_ua = train_ua | test_ua
    print(f"\n  User-Agent overlap:")
    print(f"    Train unique UAs:  {len(train_ua)}")
    print(f"    Test unique UAs:   {len(test_ua)}")
    print(f"    Overlap:           {len(overlap_ua)} / {len(total_unique_ua)} "
          f"({100*len(overlap_ua)/max(len(test_ua),1):.1f}% of test UAs in train)")

# Check source IP overlap
ip_col = 'src_ip'
if ip_col in df_loaded.columns:
    train_ip = set(df_loaded.iloc[train_idx][ip_col].dropna().unique())
    test_ip = set(df_loaded.iloc[test_idx][ip_col].dropna().unique())
    overlap_ip = train_ip & test_ip
    total_unique_ip = train_ip | test_ip
    print(f"\n  Source IP overlap:")
    print(f"    Train unique IPs:  {len(train_ip)}")
    print(f"    Test unique IPs:   {len(test_ip)}")
    print(f"    Overlap:           {len(overlap_ip)} / {len(total_unique_ip)} "
          f"({100*len(overlap_ip)/max(len(test_ip),1):.1f}% of test IPs in train)")

# Check (IP, UA) session overlap
if ua_col in df_loaded.columns and ip_col in df_loaded.columns:
    train_sessions = set(zip(
        df_loaded.iloc[train_idx][ip_col].fillna(''),
        df_loaded.iloc[train_idx][ua_col].fillna('')
    ))
    test_sessions = set(zip(
        df_loaded.iloc[test_idx][ip_col].fillna(''),
        df_loaded.iloc[test_idx][ua_col].fillna('')
    ))
    overlap_sess = train_sessions & test_sessions
    print(f"\n  Session (IP+UA) overlap:")
    print(f"    Train sessions:    {len(train_sessions)}")
    print(f"    Test sessions:     {len(test_sessions)}")
    print(f"    Overlap:           {len(overlap_sess)} / {len(test_sessions)} "
          f"({100*len(overlap_sess)/max(len(test_sessions),1):.1f}% of test sessions in train)")

print(f"\n{'=' * 60}")
print("  Done. Use these results to decide split strategy.")
print("=" * 60)
