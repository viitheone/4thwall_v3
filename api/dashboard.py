"""
Dashboard API: summary, live traffic, attack distribution, top attackers, AI status.
Serves the React dashboard; can read from decision_log when mounted, else returns mock data.
"""

import asyncio
import json
import os
from collections import defaultdict
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

router = APIRouter(prefix="/dashboard", tags=["dashboard"])

DECISION_LOG_PATH = os.environ.get("DECISION_LOG", "logs/decision_log.json")

_subscribers: set[asyncio.Event] = set()

def notify_dashboard_update() -> None:
    """Notify all connected SSE subscribers of new decisions."""
    for ev in list(_subscribers):
        ev.set()


def _get_log_fingerprint() -> tuple[float, int]:
    """Return (mtime, size) to quickly detect log file changes without reading it."""
    path = Path(DECISION_LOG_PATH)
    if not path.exists():
        return (0.0, 0)
    try:
        stat = path.stat()
        return (stat.st_mtime, stat.st_size)
    except OSError:
        return (0.0, 0)


def _read_decision_log(max_lines: int = 0) -> list[dict]:
    """Read last max_lines from decision_log.json (one JSON object per line).
    
    If max_lines is 0 or negative, all lines are returned.
    """
    path = Path(DECISION_LOG_PATH)
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        if max_lines > 0:
            lines = lines[-max_lines:]
        out = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out
    except OSError:
        return []


def _parse_request_text(text: str) -> dict:
    """Parse serialized request_text into method, path, etc."""
    result = {"method": "GET", "path": "/", "status": 200}
    for part in text.split("\n"):
        if "=" in part:
            k, v = part.split("=", 1)
            k = k.strip().upper()
            if k == "METHOD":
                result["method"] = v.strip()
            elif k == "PATH":
                result["path"] = v.strip() or "/"
            elif k == "STATUS":
                try:
                    result["status"] = int(v.strip())
                except ValueError:
                    pass
    return result


@router.get("/summary")
def get_summary():
    """Total requests, benign/malicious counts, accuracy (for dashboard)."""
    entries = _read_decision_log()
    if not entries:
        return {
            "totalRequests": 0,
            "benignRequests": 0,
            "maliciousRequests": 0,
            "modsecBlocks": 0,
            "mlBlocks": 0,
            "accuracy": 0.0,
        }
    total = len(entries)
    benign = sum(1 for e in entries if e.get("action") == "ALLOW" and (e.get("ml_score") or 0) < 0.6)
    malicious = total - benign
    modsec_blocks = sum(1 for e in entries if e.get("blocked_by") == "modsec")
    ml_blocks = sum(1 for e in entries if e.get("blocked_by") == "ml_model")
    correct = sum(
        1 for e in entries
        if (e.get("action") == "BLOCK" and (e.get("ml_score") or 0) > 0.6)
        or (e.get("action") == "ALLOW" and (e.get("ml_score") or 0) < 0.6)
    )
    accuracy = (correct / total) if total else 0.0
    return {
        "totalRequests": total,
        "benignRequests": benign,
        "maliciousRequests": malicious,
        "modsecBlocks": modsec_blocks,
        "mlBlocks": ml_blocks,
        "accuracy": round(accuracy, 2),
    }


@router.get("/live-traffic")
def get_live_traffic():
    """Last N decisions as live traffic entries for the dashboard."""
    entries = _read_decision_log(max_lines=100)
    result = []
    for i, e in enumerate(reversed(entries)):
        parsed = _parse_request_text(e.get("request_text", ""))
        action = e.get("action", "ALLOW")
        score = e.get("ml_score") or 0.0
        blocked_by = e.get("blocked_by")
        verdict = "malicious" if action == "BLOCK" or score > 0.6 else "benign"
        result.append({
            "id": f"live-{len(entries)-i}",
            "timestamp": e.get("timestamp", ""),
            "method": parsed["method"],
            "path": parsed["path"],
            "ip": "0.0.0.0",
            "verdict": verdict,
            "statusCode": parsed["status"],
            "blockedBy": blocked_by,
            "attackType": "SQLi" if "OR" in (e.get("request_text") or "") else None,
            "aiConfidence": round(score, 2),
        })
    return result


@router.get("/attack-distribution")
def get_attack_distribution():
    """Count by blocked_by source and action for chart."""
    entries = _read_decision_log()
    if not entries:
        return [{"type": "None", "count": 0}]
    by_source = defaultdict(int)
    for e in entries:
        blocked_by = e.get("blocked_by")
        if blocked_by == "modsec":
            by_source["ModSec Block"] += 1
        else:
            by_source[e.get("action", "ALLOW")] += 1
    return [{"type": k, "count": v} for k, v in sorted(by_source.items())]


@router.get("/attacks-by-hour")
def get_attacks_by_hour():
    """Count malicious/blocked by hour for chart."""
    entries = _read_decision_log()
    by_hour = defaultdict(int)
    for e in entries:
        ts = e.get("timestamp", "")[:13]
        if ts:
            by_hour[ts] += 1
    return [{"hour": h, "count": c} for h, c in sorted(by_hour.items())][-24:]


@router.get("/top-attackers")
def get_top_attackers():
    """Placeholder: no IP in decision_log; return empty or mock."""
    entries = _read_decision_log()
    blocked = [e for e in entries if e.get("action") == "BLOCK"]
    if not blocked:
        return []
    return [{"ip": "0.0.0.0", "attempts": len(blocked)}]


@router.get("/ai-status")
def get_ai_status(request: Request):
    """AI model status for dashboard; uses app state if available."""
    classifier = getattr(request.app.state, "waf_classifier", None)
    loaded = classifier is not None
    metrics = getattr(classifier, "metrics", None) if loaded else None
    
    return {
        "architecture": "DistilBERT" if loaded else "N/A",
        "version": "1.0",
        "parameters": "WAF classifier",
        "trainingProgress": 100 if loaded else 0,
        "accuracy": metrics["accuracy"] if metrics else 0.0,
        "precision": metrics["precision"] if metrics else 0.0,
        "recall": metrics["recall"] if metrics else 0.0,
        "f1": metrics["f1"] if metrics else 0.0,
    }


def _compute_dashboard_data(request: Request | None = None) -> dict:
    """Compute all dashboard stats in a single pass over the decision log."""
    entries = _read_decision_log()
    
    # 1. Summary stats
    total = len(entries)
    benign = sum(1 for e in entries if e.get("action") == "ALLOW" and (e.get("ml_score") or 0) < 0.6)
    malicious = total - benign
    modsec_blocks = sum(1 for e in entries if e.get("blocked_by") == "modsec")
    ml_blocks = sum(1 for e in entries if e.get("blocked_by") == "ml_model")
    correct = sum(
        1 for e in entries
        if (e.get("action") == "BLOCK" and (e.get("ml_score") or 0) > 0.6)
        or (e.get("action") == "ALLOW" and (e.get("ml_score") or 0) < 0.6)
    )
    accuracy = round((correct / total) if total else 0.0, 2)
    summary = {
        "totalRequests": total,
        "benignRequests": benign,
        "maliciousRequests": malicious,
        "modsecBlocks": modsec_blocks,
        "mlBlocks": ml_blocks,
        "accuracy": accuracy,
    }

    # 2. Live traffic (last 100 entries)
    live_slice = entries[-100:] if len(entries) > 100 else entries
    traffic = []
    for i, e in enumerate(reversed(live_slice)):
        parsed = _parse_request_text(e.get("request_text", ""))
        action = e.get("action", "ALLOW")
        score = e.get("ml_score") or 0.0
        blocked_by = e.get("blocked_by")
        verdict = "malicious" if action == "BLOCK" or score > 0.6 else "benign"
        traffic.append({
            "id": f"live-{len(live_slice)-i}",
            "timestamp": e.get("timestamp", ""),
            "method": parsed["method"],
            "path": parsed["path"],
            "ip": "0.0.0.0",
            "verdict": verdict,
            "statusCode": parsed["status"],
            "blockedBy": blocked_by,
            "attackType": "SQLi" if "OR" in (e.get("request_text") or "") else None,
            "aiConfidence": round(score, 2),
        })

    # 3. Attack distribution
    if not entries:
        distribution = [{"type": "None", "count": 0}]
    else:
        by_source = defaultdict(int)
        for e in entries:
            blocked_by = e.get("blocked_by")
            if blocked_by == "modsec":
                by_source["ModSec Block"] += 1
            else:
                by_source[e.get("action", "ALLOW")] += 1
        distribution = [{"type": k, "count": v} for k, v in sorted(by_source.items())]

    # 4. Attacks by hour
    by_hour = defaultdict(int)
    for e in entries:
        ts = e.get("timestamp", "")[:13]
        if ts:
            by_hour[ts] += 1
    attacks_by_hour = [{"hour": h, "count": c} for h, c in sorted(by_hour.items())][-24:]

    # 5. Top attackers
    blocked = [e for e in entries if e.get("action") == "BLOCK"]
    top_attackers = [{"ip": "0.0.0.0", "attempts": len(blocked)}] if blocked else []

    # 6. AI Model status
    classifier = getattr(request.app.state, "waf_classifier", None) if request else None
    loaded = classifier is not None
    metrics = getattr(classifier, "metrics", None) if loaded else None
    ai_status = {
        "architecture": "DistilBERT" if loaded else "N/A",
        "version": "1.0",
        "parameters": "WAF classifier",
        "trainingProgress": 100 if loaded else 0,
        "accuracy": metrics["accuracy"] if metrics else 0.0,
        "precision": metrics["precision"] if metrics else 0.0,
        "recall": metrics["recall"] if metrics else 0.0,
        "f1": metrics["f1"] if metrics else 0.0,
        "lastUpdated": "2026-02-01T00:00:00Z",
    }

    return {
        "summary": summary,
        "traffic": traffic,
        "distribution": distribution,
        "byHour": attacks_by_hour,
        "topAttackers": top_attackers,
        "aiStatus": ai_status,
    }


@router.get("/stream")
async def stream_dashboard(request: Request):
    """Server-Sent Events endpoint streaming real-time dashboard updates."""
    async def event_generator():
        ev = asyncio.Event()
        _subscribers.add(ev)
        last_fingerprint = None
        try:
            # Immediate snapshot on connection
            data = _compute_dashboard_data(request)
            last_fingerprint = _get_log_fingerprint()
            yield f"data: {json.dumps(data)}\n\n"

            while True:
                if await request.is_disconnected():
                    break
                try:
                    await asyncio.wait_for(ev.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    pass
                ev.clear()

                if await request.is_disconnected():
                    break

                current_fingerprint = _get_log_fingerprint()
                if current_fingerprint != last_fingerprint:
                    last_fingerprint = current_fingerprint
                    data = _compute_dashboard_data(request)
                    yield f"data: {json.dumps(data)}\n\n"
                else:
                    yield ": keepalive\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            _subscribers.discard(ev)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


class FlagFPRequest(BaseModel):
    id: str
    timestamp: str
    method: str
    path: str
    statusCode: int

@router.post("/flag-fp")
def flag_false_positive(req: FlagFPRequest, request: Request):
    """Flag an entry as a False Positive, adding it to retrain logs."""
    RETRAIN_LOG_PATH = os.environ.get("RETRAIN_LOG", "logs/retrain_log.json")
    
    entries = _read_decision_log(max_lines=500)
    matched_entry = None
    for e in reversed(entries):
        if e.get("timestamp") == req.timestamp:
            matched_entry = e
            break
            
    if not matched_entry:
        row = {"method": req.method, "path": req.path, "status": req.statusCode, "user_agent": "", "request_time": 0.0}
        req_text = "\n".join(f"{k.upper()}={v}" for k, v in row.items())
    else:
        req_text = matched_entry.get("request_text", "")
        
    os.makedirs(os.path.dirname(RETRAIN_LOG_PATH), exist_ok=True)
    with open(RETRAIN_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "request_text": req_text,
            "ml_score": matched_entry.get("ml_score", 1.0) if matched_entry else 1.0,
            "corrected_label": "benign",
            "source": "ui_flag",
            "timestamp": req.timestamp,
            "action_taken": matched_entry.get("action", "BLOCK") if matched_entry else "BLOCK"
        }) + "\n")
        
    return {"status": "success"}
