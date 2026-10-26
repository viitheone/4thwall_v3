"""
Model v2.0 — Preprocessing
Convert raw HTTP request text into tokenizable strings and multi-label vectors.
"""

import re
import numpy as np
import pandas as pd

from config import CAPEC_LABELS, NUM_LABELS


def clean_http_text(text: str) -> str:
    """Clean and normalize raw HTTP request text for tokenization.

    - Lowercases everything
    - Normalizes excessive whitespace
    - Keeps URL-relevant chars (/, ?, =, &, %, etc.)
    - Truncates extremely long payloads
    """
    if not isinstance(text, str):
        return ""
    text = text.strip().lower()
    # Collapse multiple whitespace/newlines into single space
    text = re.sub(r'\s+', ' ', text)
    # Remove null bytes and control chars (keep printable + common URL chars)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    return text


def serialize_http_request(row: dict) -> str:
    """Convert a row/dict into a single text string for the model.

    Concatenates available HTTP fields into one sequence:
    METHOD PATH?QUERY HTTP/VERSION\\nHeader1: value\\nHeader2: value\\n\\nBODY

    Adapts to whatever columns are available in the dataset.
    """
    parts = []

    # Request line
    method = str(row.get('method', row.get('Method', ''))).upper()
    path = str(row.get('path', row.get('Path', row.get('url', row.get('URL', '/')))))
    query = str(row.get('query', row.get('Query', '')))
    if query and query != 'nan':
        path = f"{path}?{query}"
    if method:
        parts.append(f"{method} {path}")

    # Headers (if available)
    for hdr_key in ['host', 'Host', 'user_agent', 'User-Agent', 'user-agent',
                     'content_type', 'Content-Type', 'content-type',
                     'referer', 'Referer', 'cookie', 'Cookie']:
        val = row.get(hdr_key, '')
        if val and str(val) != 'nan':
            parts.append(f"{hdr_key}: {val}")

    # Body (if available)
    body = row.get('body', row.get('Body', row.get('payload', '')))
    if body and str(body) != 'nan':
        parts.append(f"\n{body}")

    # If we have a single 'Full_HTTP_Request' column, use it directly
    full_req = row.get('Full_HTTP_Request', row.get('full_http_request', ''))
    if full_req and str(full_req) != 'nan':
        return clean_http_text(str(full_req))

    return clean_http_text(' '.join(parts))


def serialize_srbh_row(row: dict, stats: dict = None) -> str:
    """Serialize a SR-BH 2020 dataset row using structured delimiter tokens.

    Format: [METHOD] POST [PATH] /blog/... [UA] Mozilla/5.0... [BODY] payload...

    Uses [FIELD] delimiters so DistilBERT can tell fields apart, rather than
    raw HTTP format which would require the model to learn HTTP syntax.

    Args:
        row: dict of column values from the SR-BH 2020 dataset
        stats: optional dict to track truncation counts (mutated in-place)

    Returns:
        Cleaned, delimited text string ready for tokenization.
    """
    from config import MAX_BODY_CHARS, INCLUDE_COOKIE

    def _safe(val):
        """Return empty string for NaN/None values."""
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return ""
        s = str(val).strip()
        return "" if s.lower() == "nan" else s

    parts = []

    # [METHOD] — critical for HTTP Verb Tampering (CAPEC-274)
    method = _safe(row.get('request_http_method', ''))
    if method:
        parts.append(f"[METHOD] {method}")

    # [PATH] — core signal for SQLi, path traversal, etc.
    path = _safe(row.get('request_http_request', ''))
    if path:
        parts.append(f"[PATH] {path}")

    # [HOST]
    host = _safe(row.get('request_host', ''))
    if host:
        parts.append(f"[HOST] {host}")

    # [UA] — detects Scanning (CAPEC-310), fingerprinting
    ua = _safe(row.get('request_user_agent', ''))
    if ua:
        parts.append(f"[UA] {ua}")

    # [REF] — detects Data Source Forgery (CAPEC-194)
    referer = _safe(row.get('request_referer', ''))
    if referer:
        parts.append(f"[REF] {referer}")

    # [CT] — detects Input Data Manipulation (CAPEC-153)
    ct = _safe(row.get('request_content_type', ''))
    if ct:
        parts.append(f"[CT] {ct}")

    # [ORIGIN]
    origin = _safe(row.get('request_origin', ''))
    if origin:
        parts.append(f"[ORIGIN] {origin}")

    # [COOKIE] — excluded by default (usually just session tokens / random noise).
    # Enable via config.INCLUDE_COOKIE for ablation study.
    if INCLUDE_COOKIE:
        cookie = _safe(row.get('request_cookie', ''))
        if cookie:
            parts.append(f"[COOKIE] {cookie}")

    # [BODY] — critical for OS Command Injection (CAPEC-88), Code Injection
    # Truncated to MAX_BODY_CHARS to stay within token budget.
    body = _safe(row.get('request_body', ''))
    if body:
        was_truncated = len(body) > MAX_BODY_CHARS
        if was_truncated:
            body = body[:MAX_BODY_CHARS]
            if stats is not None:
                stats["body_truncated"] = stats.get("body_truncated", 0) + 1
        parts.append(f"[BODY] {body}")

    if stats is not None:
        stats["total"] = stats.get("total", 0) + 1

    return clean_http_text(' '.join(parts))


def encode_labels(row: dict, label_columns: list = None) -> np.ndarray:
    """Convert a row's label columns into a binary multi-label vector.

    Args:
        row: dict-like with label columns
        label_columns: list of column names to use as labels.
                       If None, uses CAPEC_LABELS from config.

    Returns:
        np.ndarray of shape (NUM_LABELS,) with 0s and 1s
    """
    if label_columns is None:
        label_columns = CAPEC_LABELS

    labels = np.zeros(len(label_columns), dtype=np.float32)
    for i, col in enumerate(label_columns):
        val = row.get(col, 0)
        if isinstance(val, str):
            val = val.strip().lower()
            labels[i] = 1.0 if val in ('1', 'true', 'yes') else 0.0
        else:
            labels[i] = 1.0 if val == 1 else 0.0
    return labels


def detect_label_columns(df: pd.DataFrame) -> list:
    """Auto-detect which columns in the DataFrame are CAPEC label columns.

    Looks for columns matching CAPEC_LABELS names (exact match first).
    """
    found = []

    # Try exact match against configured CAPEC_LABELS
    for col in CAPEC_LABELS:
        if col in df.columns:
            found.append(col)

    if found:
        return found

    # Fallback: look for columns with pattern "NUMBER - Name" (CAPEC format)
    import re
    capec_pattern = re.compile(r'^\d+\s*-\s*.+$')
    capec_cols = [c for c in df.columns if capec_pattern.match(c)]
    if capec_cols:
        print(f"  Found {len(capec_cols)} CAPEC-format columns by pattern matching")
        return capec_cols

    # Last resort: binary columns (excluding known non-label columns)
    exclude = {'index', 'id', 'timestamp', 'src_ip', 'src_port', 'dst_ip', 'dst_port',
               'request_do_not_track', 'request_connection'}
    binary_cols = [c for c in df.columns
                   if c.lower() not in exclude
                   and df[c].dropna().isin([0, 1]).all()
                   and df[c].nunique() == 2]
    if binary_cols:
        print(f"  Warning: No CAPEC columns found by name. "
              f"Found {len(binary_cols)} binary columns: {binary_cols[:5]}...")
        return binary_cols

    return []


def detect_text_column(df: pd.DataFrame) -> str:
    """Auto-detect which column contains the HTTP request text."""
    from config import TEXT_COLUMN

    # Try configured column first
    if TEXT_COLUMN in df.columns:
        return TEXT_COLUMN

    candidates = [
        'request_http_request', 'Full_HTTP_Request', 'full_http_request',
        'request', 'Request', 'http_request',
        'text', 'Text', 'payload', 'Payload',
        'url', 'URL', 'path', 'Path',
    ]
    for col in candidates:
        if col in df.columns:
            return col

    # Fallback: find the longest string column
    str_cols = df.select_dtypes(include='object').columns
    if len(str_cols) > 0:
        avg_lens = {c: df[c].astype(str).str.len().mean() for c in str_cols}
        best = max(avg_lens, key=avg_lens.get)
        print(f"  Warning: No known text column found. Using '{best}' (avg len={avg_lens[best]:.0f})")
        return best

    raise ValueError("Cannot find any text column in the dataset. "
                     "Expected 'Full_HTTP_Request' or similar.")
