"""
Model v2.0 — Dataset
PyTorch Dataset class + time-based train/val/test split.
Handles label merging and serialization stats.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split, GroupShuffleSplit

from config import (
    BASE_MODEL, MAX_LENGTH, BATCH_SIZE, NUM_LABELS, CAPEC_LABELS,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED,
    LABEL_MERGE_MAP, TIME_BASED_SPLIT, TIMESTAMP_COLUMN,
    TIMESTAMP_FORMAT, SESSION_KEY_COLUMN,
)
from preprocess import (
    serialize_http_request, encode_labels,
    detect_label_columns, detect_text_column, clean_http_text,
    serialize_srbh_row,
)


class WAFMultiLabelDataset(Dataset):
    """PyTorch Dataset for multi-label WAF classification.

    Pre-tokenizes all texts at construction time so __getitem__ is pure
    tensor indexing — no CPU tokenizer bottleneck during training.
    """

    def __init__(self, texts: list, labels: np.ndarray, tokenizer, max_length: int):
        """
        Args:
            texts: list of preprocessed HTTP request strings
            labels: np.ndarray of shape (N, NUM_LABELS) with 0/1 values
            tokenizer: HuggingFace tokenizer
            max_length: max token length
        """
        from tqdm import tqdm as tqdm_bar

        # Pre-tokenize everything once (biggest speedup)
        print(f"    Pre-tokenizing {len(texts)} samples (max_length={max_length})...")
        batch_size = 4096  # tokenize in large chunks for speed
        all_input_ids = []
        all_attention_masks = []

        for i in tqdm_bar(range(0, len(texts), batch_size),
                          desc="    Tokenizing", bar_format='{l_bar}{bar:30}{r_bar}',
                          unit='chunk'):
            chunk = texts[i:i + batch_size]
            enc = tokenizer(
                chunk,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            all_input_ids.append(enc["input_ids"])
            all_attention_masks.append(enc["attention_mask"])

        self.input_ids = torch.cat(all_input_ids, dim=0)
        self.attention_masks = torch.cat(all_attention_masks, dim=0)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        print(f"    Done. Tensor shapes: input_ids={list(self.input_ids.shape)}, "
              f"labels={list(self.labels.shape)}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "labels": self.labels[idx],
        }


def time_based_split(df, texts, labels, train_ratio, val_ratio, test_ratio, ts_col):
    """Split data chronologically to prevent same-attacker session leakage.

    Sorts by timestamp (Apache CLF format) and splits into contiguous
    time windows. Earlier traffic = train, later traffic = test.

    Returns None if timestamps can't be parsed (caller should fall back).
    """
    texts = np.array(texts, dtype=object)
    labels = np.array(labels)

    # Parse timestamps with explicit CLF format
    print(f"  Parsing timestamps (format: {TIMESTAMP_FORMAT})...")
    timestamps = pd.to_datetime(df[ts_col], format=TIMESTAMP_FORMAT, errors='coerce')

    # Safety check: if too many NaT, format is wrong
    nat_count = timestamps.isna().sum()
    nat_ratio = nat_count / len(df)
    print(f"  Parsed: {len(df) - nat_count}/{len(df)} valid ({100*(1-nat_ratio):.1f}%)")

    if nat_ratio > 0.2:
        print(f"  WARNING: {nat_ratio:.0%} NaT -- timestamp format may be wrong.")
        return None

    # Sort by timestamp
    sort_idx = timestamps.argsort().values
    texts = texts[sort_idx]
    labels = labels[sort_idx]
    df_sorted = df.iloc[sort_idx].reset_index(drop=True)
    sorted_ts = timestamps.iloc[sort_idx].reset_index(drop=True)

    n = len(texts)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_texts = texts[:train_end].tolist()
    train_labels = labels[:train_end]
    val_texts = texts[train_end:val_end].tolist()
    val_labels = labels[train_end:val_end]
    test_texts = texts[val_end:].tolist()
    test_labels = labels[val_end:]

    # Report time ranges
    print(f"  Chronological split:")
    print(f"    Train: {sorted_ts.iloc[0]} -> {sorted_ts.iloc[train_end-1]}")
    print(f"    Val:   {sorted_ts.iloc[train_end]} -> {sorted_ts.iloc[val_end-1]}")
    print(f"    Test:  {sorted_ts.iloc[val_end]} -> {sorted_ts.iloc[-1]}")

    # Sanity check: verify attacks exist in all splits
    for split_name, split_labels in [("train", train_labels), ("val", val_labels), ("test", test_labels)]:
        attack_count = int(split_labels[:, 1:].sum())
        if attack_count == 0:
            print(f"  CRITICAL: {split_name} split has ZERO attack samples!")
            return None

    # Report UA overlap between train and test (leakage check)
    ua_col = SESSION_KEY_COLUMN
    if ua_col in df_sorted.columns:
        train_ua = set(df_sorted.iloc[:train_end][ua_col].dropna().unique())
        test_ua = set(df_sorted.iloc[val_end:][ua_col].dropna().unique())
        overlap = train_ua & test_ua
        print(f"\n  Leakage check (User-Agent overlap):")
        print(f"    Train unique UAs: {len(train_ua)}")
        print(f"    Test unique UAs:  {len(test_ua)}")
        print(f"    Overlap:          {len(overlap)}/{len(test_ua)} "
              f"({100*len(overlap)/max(len(test_ua),1):.1f}% of test UAs seen in train)")

    # Per-split class distribution
    print(f"\n  Per-split attack distribution:")
    for split_name, split_labels in [("train", train_labels), ("val", val_labels), ("test", test_labels)]:
        attack_total = int(split_labels[:, 1:].sum())
        normal_total = int(split_labels[:, 0].sum())
        print(f"    {split_name:5s}: {len(split_labels)} samples, "
              f"{normal_total} normal, {attack_total} attack labels")

    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels


def stratified_multilabel_split(texts, labels, train_ratio, val_ratio, test_ratio, seed):
    """Split data into train/val/test with approximate stratification for multi-label.

    WARNING: For honeypot data, use time_based_split instead to prevent
    same-attacker session leakage between splits.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    n = len(texts)
    texts = np.array(texts, dtype=object)
    labels = np.array(labels)

    # Create a composite stratification key from top-3 rarest labels
    label_sums = labels.sum(axis=0)
    sorted_by_freq = np.argsort(label_sums)
    strat_cols = [i for i in sorted_by_freq if label_sums[i] > 0][:3]
    strat_key = ["".join(str(int(labels[i, c])) for c in strat_cols) for i in range(n)]

    # First split: train vs (val+test)
    val_test_ratio = val_ratio + test_ratio
    try:
        train_idx, valtest_idx = train_test_split(
            np.arange(n), test_size=val_test_ratio,
            random_state=seed, stratify=strat_key,
        )
    except ValueError:
        print("  Warning: Stratified split failed, falling back to random split")
        train_idx, valtest_idx = train_test_split(
            np.arange(n), test_size=val_test_ratio, random_state=seed,
        )

    # Second split: val vs test
    relative_test = test_ratio / val_test_ratio
    valtest_strat = [strat_key[i] for i in valtest_idx]
    try:
        val_idx, test_idx = train_test_split(
            valtest_idx, test_size=relative_test,
            random_state=seed, stratify=valtest_strat,
        )
    except ValueError:
        val_idx, test_idx = train_test_split(
            valtest_idx, test_size=relative_test, random_state=seed,
        )

    return (
        texts[train_idx].tolist(), labels[train_idx],
        texts[val_idx].tolist(), labels[val_idx],
        texts[test_idx].tolist(), labels[test_idx],
    )


def group_split(df, texts, labels, train_ratio, val_ratio, test_ratio, seed):
    """Split data by user-agent group: 0% UA overlap between splits.

    Uses sklearn GroupShuffleSplit to ensure every row from the same
    user-agent session goes entirely into train OR test, never both.
    This eliminates session leakage from honeypot datasets where the
    same bot hits the honeypot repeatedly with an identical UA.

    Note: some attack classes may have zero samples in test if all
    instances of that class come from a small number of UAs.
    """
    texts = np.array(texts, dtype=object)
    labels = np.array(labels)

    # Group by user-agent
    ua_col = SESSION_KEY_COLUMN
    if ua_col not in df.columns:
        print(f"  WARNING: '{ua_col}' column not found. Cannot do group split.")
        return None

    groups = df[ua_col].fillna('__MISSING__').values

    # First split: train vs (val+test)
    val_test_ratio = val_ratio + test_ratio
    gss1 = GroupShuffleSplit(n_splits=1, test_size=val_test_ratio, random_state=seed)
    train_idx, valtest_idx = next(gss1.split(texts, labels, groups))

    # Second split: val vs test
    relative_test = test_ratio / val_test_ratio
    valtest_groups = groups[valtest_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=relative_test, random_state=seed)
    val_sub_idx, test_sub_idx = next(gss2.split(
        texts[valtest_idx], labels[valtest_idx], valtest_groups
    ))
    val_idx = valtest_idx[val_sub_idx]
    test_idx = valtest_idx[test_sub_idx]

    # Report overlap
    train_uas = set(groups[train_idx])
    test_uas = set(groups[test_idx])
    val_uas = set(groups[val_idx])
    overlap_test = train_uas & test_uas
    overlap_val = train_uas & val_uas

    print(f"  Group split (by {ua_col}):")
    print(f"    Train: {len(train_idx)} rows, {len(train_uas)} unique groups")
    print(f"    Val:   {len(val_idx)} rows, {len(val_uas)} unique groups")
    print(f"    Test:  {len(test_idx)} rows, {len(test_uas)} unique groups")
    print(f"    Train-Test UA overlap: {len(overlap_test)} ({100*len(overlap_test)/max(len(test_uas),1):.1f}%)")
    print(f"    Train-Val UA overlap:  {len(overlap_val)} ({100*len(overlap_val)/max(len(val_uas),1):.1f}%)")

    # Per-class coverage warning
    test_labels = labels[test_idx]
    zero_classes = []
    for j in range(labels.shape[1]):
        if test_labels[:, j].sum() == 0:
            zero_classes.append(j)
    if zero_classes:
        print(f"    WARNING: {len(zero_classes)} classes have ZERO test samples (cannot evaluate these)")

    return (
        texts[train_idx].tolist(), labels[train_idx],
        texts[val_idx].tolist(), labels[val_idx],
        texts[test_idx].tolist(), labels[test_idx],
    )


def merge_labels(df, label_columns):
    """Merge labels according to LABEL_MERGE_MAP.

    E.g. "248 - Command Injection" (1 sample) gets merged into
    "88 - OS Command Injection" (parent category in CAPEC hierarchy).

    Returns:
        df: DataFrame with merged label columns
        merged_label_columns: updated list of label column names
    """
    if not LABEL_MERGE_MAP:
        return df, label_columns

    merged_cols = list(label_columns)
    for src_col, dst_col in LABEL_MERGE_MAP.items():
        if src_col in df.columns and dst_col in df.columns:
            # OR-merge: if either source or dest is 1, result is 1
            df[dst_col] = (df[dst_col].fillna(0).astype(int) |
                           df[src_col].fillna(0).astype(int)).astype(int)
            df = df.drop(columns=[src_col])
            if src_col in merged_cols:
                merged_cols.remove(src_col)

            src_count = int(df[dst_col].sum())
            print(f"  Merged '{src_col}' -> '{dst_col}' (combined count: {src_count})")
        elif src_col in merged_cols:
            # Source column exists in label list but not in df — just remove it
            merged_cols.remove(src_col)
            print(f"  Dropped '{src_col}' (not found in dataset)")

    return df, merged_cols


def load_dataset(csv_path: str, sep=None):
    """Load and preprocess the SR-BH 2020 dataset from CSV/TSV.

    Returns:
        texts: list of cleaned HTTP request strings
        labels: np.ndarray of shape (N, NUM_LABELS)
        label_columns: list of actual column names used
        df: the loaded DataFrame (needed for time-based split)
    """
    print(f"\nLoading dataset from {csv_path}...")

    # Auto-detect separator
    if sep is None:
        with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
            first_line = f.readline()
            if '\t' in first_line and ',' not in first_line:
                sep = '\t'
            else:
                sep = ','

    df = pd.read_csv(csv_path, sep=sep, encoding='utf-8', on_bad_lines='skip', low_memory=False)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"  Columns: {list(df.columns)}")

    # Detect label columns BEFORE merging
    label_columns = detect_label_columns(df)
    text_col = detect_text_column(df)

    print(f"  Text column: '{text_col}'")
    print(f"  Label columns ({len(label_columns)}): {label_columns}")

    if len(label_columns) == 0:
        raise ValueError("No label columns detected. Check dataset format.")

    # Merge labels (e.g. CAPEC-248 → CAPEC-88)
    df, label_columns = merge_labels(df, label_columns)
    print(f"  Label columns after merge ({len(label_columns)}): {label_columns}")

    # Drop rows with missing text
    df = df.dropna(subset=[text_col]).reset_index(drop=True)
    print(f"  After dropping NaN text: {len(df)} rows")

    # Serialize text — full HTTP row with structured delimiters
    from tqdm import tqdm as tqdm_bar
    use_full_row = 'request_http_method' in df.columns

    if use_full_row:
        from config import INCLUDE_COOKIE, MAX_BODY_CHARS
        print(f"  Using structured serialization: [METHOD] [PATH] [UA] [CT] [BODY]...")
        print(f"    Cookie included: {INCLUDE_COOKIE}")
        print(f"    Body truncation: {MAX_BODY_CHARS} chars max")

        stats = {"total": 0, "body_truncated": 0}
        texts = []
        for _, row in tqdm_bar(df.iterrows(), total=len(df), desc="  Serializing rows",
                                bar_format='{l_bar}{bar:30}{r_bar}', unit='row'):
            texts.append(serialize_srbh_row(row.to_dict(), stats=stats))

        # Log truncation stats (reproducibility detail)
        trunc_pct = 100.0 * stats["body_truncated"] / max(stats["total"], 1)
        print(f"  Body truncation: {stats['body_truncated']}/{stats['total']} "
              f"({trunc_pct:.1f}%) samples had body truncated to {MAX_BODY_CHARS} chars")
    else:
        print(f"  Using single text column: '{text_col}'")
        texts = []
        for _, row in tqdm_bar(df.iterrows(), total=len(df), desc="  Extracting text",
                                bar_format='{l_bar}{bar:30}{r_bar}', unit='row'):
            texts.append(clean_http_text(str(row[text_col])))

    # Extract labels
    labels = np.zeros((len(df), len(label_columns)), dtype=np.float32)
    for i, (_, row) in enumerate(tqdm_bar(df.iterrows(), total=len(df), desc="  Encoding labels",
                                           bar_format='{l_bar}{bar:30}{r_bar}', unit='row')):
        labels[i] = encode_labels(row, label_columns)

    # Report class distribution
    print(f"\n  Class distribution:")
    for j, col in enumerate(label_columns):
        pos = int(labels[:, j].sum())
        pct = 100.0 * pos / len(labels)
        print(f"    {col:.<25s} {pos:>7d} ({pct:5.1f}%)")

    # Multi-label stats
    labels_per_sample = labels.sum(axis=1)
    print(f"\n  Labels per sample: mean={labels_per_sample.mean():.2f}, "
          f"max={int(labels_per_sample.max())}, "
          f"multi-label samples={int((labels_per_sample > 1).sum())}")

    return texts, labels, label_columns, df


def create_dataloaders(csv_path: str, batch_size: int = BATCH_SIZE, sep=None, split_mode=None):
    """Full pipeline: load data -> merge labels -> split -> tokenize -> DataLoaders.

    Args:
        split_mode: 'stratified', 'group', 'time', or None (auto-detect).

    Returns:
        train_loader, val_loader, test_loader, label_columns, class_weights
    """
    texts, labels, label_columns, df = load_dataset(csv_path, sep=sep)

    # Choose split strategy
    print(f"\n  Splitting: {TRAIN_RATIO:.0%} train / {VAL_RATIO:.0%} val / {TEST_RATIO:.0%} test")

    split_used = "unknown"

    if split_mode == 'group':
        # Forced group split (for honest evaluation)
        result = group_split(df, texts, labels, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED)
        if result is not None:
            (train_texts, train_labels,
             val_texts, val_labels,
             test_texts, test_labels) = result
            split_used = "group (by UA)"
        else:
            print(f"  Group split failed, falling back to stratified.")
            (train_texts, train_labels,
             val_texts, val_labels,
             test_texts, test_labels) = stratified_multilabel_split(
                texts, labels, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED,
            )
            split_used = "stratified (fallback)"
    elif split_mode == 'stratified' or (split_mode is None and not TIME_BASED_SPLIT):
        (train_texts, train_labels,
         val_texts, val_labels,
         test_texts, test_labels) = stratified_multilabel_split(
            texts, labels, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED,
        )
        split_used = "stratified"
    else:
        # Auto: try time-based, fall back to stratified
        if TIMESTAMP_COLUMN in df.columns:
            result = time_based_split(
                df, texts, labels, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, TIMESTAMP_COLUMN,
            )
            if result is not None:
                (train_texts, train_labels,
                 val_texts, val_labels,
                 test_texts, test_labels) = result
                split_used = "chronological"
            else:
                print(f"  Falling back to stratified split.")
                (train_texts, train_labels,
                 val_texts, val_labels,
                 test_texts, test_labels) = stratified_multilabel_split(
                    texts, labels, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED,
                )
                split_used = "stratified (fallback)"
        else:
            print(f"  Warning: '{TIMESTAMP_COLUMN}' column not found.")
            (train_texts, train_labels,
             val_texts, val_labels,
             test_texts, test_labels) = stratified_multilabel_split(
                texts, labels, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED,
            )
            split_used = "stratified"

    print(f"  Split method: {split_used}")
    print(f"  Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")

    # Compute per-class positive weights for BCEWithLogitsLoss
    # pos_weight = num_negatives / num_positives per class
    train_pos = train_labels.sum(axis=0)
    train_neg = len(train_labels) - train_pos
    # Avoid division by zero; cap weight at 50
    pos_weight = np.clip(train_neg / np.maximum(train_pos, 1), 1.0, 50.0)
    print(f"\n  Class pos_weights (for imbalance): {dict(zip(label_columns, pos_weight.round(1).tolist()))}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # Create datasets
    train_ds = WAFMultiLabelDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_ds = WAFMultiLabelDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)
    test_ds = WAFMultiLabelDataset(test_texts, test_labels, tokenizer, MAX_LENGTH)

    # DataLoaders (pin_memory=True for faster CPU→GPU transfer)
    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=pin)

    class_weights = torch.tensor(pos_weight, dtype=torch.float32)

    return train_loader, val_loader, test_loader, label_columns, class_weights
