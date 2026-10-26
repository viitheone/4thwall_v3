"""
Request serialization and preprocessing protocol for 4thwall_v3 WAF.
Follows the canonical specification from MIGRATION_v3.md.
"""

import re
from typing import Any, Mapping
from ml.config import DEFAULT_UA, MAX_BODY_CHARS


def serialize_request(
    method: Any = "GET",
    path: str = "/",
    ua: str = "",
    ct: str = "",
    body: str = "",
    max_body_chars: int = MAX_BODY_CHARS
) -> str:
    """
    Canonical request serializer for Frozen Model v2.1b / HuggingFace WAF.
    
    Produces formatted markup tokens:
    [METHOD] {m} [PATH] {p} [UA] {ua} [CT] {ct} [BODY] {body}

    Supports both keyword-based invocation and legacy dictionary row objects.
    """
    # Handle dict or row-like object passed as first positional parameter
    if isinstance(method, Mapping) or hasattr(method, "to_dict") or hasattr(method, "get"):
        row = method.to_dict() if hasattr(method, "to_dict") else dict(method)
        m = row.get("method") or row.get("METHOD") or "GET"
        
        p = row.get("path") or row.get("url") or row.get("uri") or "/"
        query = row.get("query") or row.get("query_string") or row.get("args") or ""
        if query and "?" not in str(p):
            p = f"{p}?{query}"
            
        u = row.get("user_agent") or row.get("ua") or row.get("user-agent") or ""
        c = row.get("content_type") or row.get("ct") or row.get("content-type") or ""
        b = row.get("body") or row.get("payload") or row.get("data") or ""
        return serialize_request(method=m, path=p, ua=u, ct=c, body=b, max_body_chars=max_body_chars)

    parts = []

    # 1. Method: uppercase, strip whitespace, default GET
    m = str(method).strip().upper() if method and str(method).lower() not in ('nan', 'none', '') else "GET"
    parts.append(f"[METHOD] {m}")

    # 2. Path: preserve original query params and transport encoding
    p = str(path).strip() if path and str(path).lower() not in ('nan', 'none', '') else "/"
    if not p.startswith("/"):
        p = "/" + p
    parts.append(f"[PATH] {p}")

    # 3. User-Agent: MUST default to standard browser if empty (prevents scanner heuristic false alarms)
    ua_str = str(ua).strip() if ua and str(ua).lower() not in ('nan', 'none', '') else DEFAULT_UA
    parts.append(f"[UA] {ua_str}")

    # 4. Content-Type (optional)
    if ct and str(ct).lower() not in ('nan', 'none', ''):
        parts.append(f"[CT] {str(ct).strip()}")

    # 5. Body (truncated to 500 characters to prevent DistilBERT token starvation)
    if body and str(body).lower() not in ('nan', 'none', ''):
        b_str = str(body).strip()[:max_body_chars]
        parts.append(f"[BODY] {b_str}")

    return " ".join(parts)


def parse_serialized_request(text: str) -> dict:
    """
    Parse a serialized request string back into its constituent components.
    Supports both canonical token format:
      [METHOD] GET [PATH] / [UA] Mozilla/... [CT] text/html [BODY] ...
    and legacy key=value lines.
    """
    result = {
        "method": "GET",
        "path": "/",
        "user_agent": DEFAULT_UA,
        "content_type": "",
        "body": "",
        "status": 200,
    }
    if not text:
        return result

    # Check for canonical markup tokens [TAG]
    if "[METHOD]" in text or "[PATH]" in text:
        pattern = r"\[(METHOD|PATH|UA|CT|BODY)\]\s*([^\[]*)"
        matches = re.findall(pattern, text)
        for tag, val in matches:
            val = val.strip()
            if tag == "METHOD":
                result["method"] = val.upper()
            elif tag == "PATH":
                result["path"] = val
            elif tag == "UA":
                result["user_agent"] = val
            elif tag == "CT":
                result["content_type"] = val
            elif tag == "BODY":
                result["body"] = val
        return result

    # Legacy fallback: METHOD=GET\nPATH=...
    for line in text.split("\n"):
        if "=" in line:
            k, v = line.split("=", 1)
            k = k.strip().upper()
            v = v.strip()
            if k == "METHOD":
                result["method"] = v.upper()
            elif k == "PATH":
                result["path"] = v or "/"
            elif k in ("UA", "USER_AGENT"):
                result["user_agent"] = v or DEFAULT_UA
            elif k in ("STATUS", "STATUS_CODE"):
                try:
                    result["status"] = int(v)
                except ValueError:
                    pass
            elif k == "BODY":
                result["body"] = v

    return result


def clean_text(text: str) -> str:
    """Legacy helper: safe strip without regex character mangling."""
    if text is None or (isinstance(text, float) and str(text) == "nan"):
        return ""
    return str(text).strip()


def truncate_field(text: str, max_len: int = 200) -> str:
    """Truncate field string to max_len."""
    if text is None or (isinstance(text, float) and str(text) == "nan"):
        return ""
    s = str(text).strip()
    return s[:max_len] if len(s) > max_len else s
