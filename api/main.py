import logging
import os
import json
import time
import asyncio
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx

from api.dashboard import router as dashboard_router, notify_dashboard_update
from api.schemas import (
    BatchScoreRequest,
    BatchScoreResponse,
    ScoreRequest,
    ScoreResponse,
    InspectRequest,
    InspectResponse,
)
from ml.config import HF_MODEL_REPO
from ml.infer import WAFClassifier
from ml.preprocess import serialize_request
from policy.decision import make_decision

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

classifier: Optional[WAFClassifier] = None
http_client = httpx.AsyncClient(timeout=30.0)

RETRAIN_LOG_PATH = os.environ.get("RETRAIN_LOG", "logs/retrain_log.json")
ACCESS_LOG_PATH = os.environ.get("ACCESS_LOG_PATH", "logs/access.log")
OFFSET_PATH = os.environ.get("OFFSET_PATH", "logs/.waf_log_worker_offset")
DECISION_LOG_PATH = os.environ.get("DECISION_LOG", "logs/decision_log.json")
DVWA_URL = os.environ.get("DVWA_URL", "http://dvwa:80")


def append_decision(entry: dict) -> None:
    os.makedirs(os.path.dirname(DECISION_LOG_PATH), exist_ok=True)
    with open(DECISION_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    notify_dashboard_update()


async def modsec_block_logger():
    """Background task: tail Nginx access.log for ModSec 403 blocks."""
    while True:
        await asyncio.sleep(60)
        if not os.path.exists(ACCESS_LOG_PATH):
            continue
        try:
            offset = 0
            if os.path.exists(OFFSET_PATH):
                with open(OFFSET_PATH, "r") as f:
                    offset = int(f.read().strip() or "0")
            file_size = os.path.getsize(ACCESS_LOG_PATH)
            if file_size < offset:
                offset = 0
            if file_size == offset:
                continue
            
            os.makedirs(os.path.dirname(OFFSET_PATH), exist_ok=True)
            with open(ACCESS_LOG_PATH, "r", encoding="utf-8", errors="ignore") as f:
                f.seek(offset)
                lines = f.readlines()
                new_offset = f.tell()
                
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    status = data.get("status")
                    
                    if status == 403 and data.get("request_time", 1.0) < 0.1:
                        query = data.get("query", "")
                        uri = data.get("path", "/")
                        if query and "?" not in uri:
                            uri += "?" + query
                        
                        method = data.get("method", "GET")
                        ua = data.get("user_agent", "")
                        req_text = serialize_request(method=method, path=uri, ua=ua)
                        
                        ml_res = None
                        ml_score = 0.0
                        if classifier is not None:
                            try:
                                ml_res = classifier.predict(req_text)
                                ml_score = ml_res.get("score", 0.0)
                            except Exception:
                                pass
                        
                        append_decision({
                            "timestamp": data.get("time", datetime.utcnow().isoformat() + "Z"),
                            "request_text": req_text,
                            "modsecurity_blocked": True,
                            "ml_score": ml_score,
                            "action": "BLOCK",
                            "reason": "ModSecurity CRS rule triggered",
                            "blocked_by": "modsec",
                            "firing_labels": ml_res.get("firing_labels", []) if ml_res else [],
                            "primary_attack": ml_res.get("primary_attack") if ml_res else None,
                        })
                        
                        if ml_score < 0.6:
                            os.makedirs(os.path.dirname(RETRAIN_LOG_PATH), exist_ok=True)
                            with open(RETRAIN_LOG_PATH, "a", encoding="utf-8") as rf:
                                rf.write(json.dumps({
                                    "request_text": req_text,
                                    "ml_score": ml_score,
                                    "corrected_label": "malicious",
                                    "source": "modsec_sync",
                                    "timestamp": data.get("time", ""),
                                    "action_taken": "BLOCK"
                                }) + "\n")
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    logger.error(f"Error processing access log line: {e}")
                    
            with open(OFFSET_PATH, "w") as f:
                f.write(str(new_offset))
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in modsec block logger: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global classifier
    model_path = os.environ.get("MODEL_PATH", HF_MODEL_REPO)
    try:
        classifier = WAFClassifier(model_path)
        logger.info("WAFClassifier loaded from %s", model_path)
    except Exception as e:
        classifier = None
        logger.warning("Failed to load model from %s: %s  — /score endpoints will return 503", model_path, e)
    app.state.waf_classifier = classifier

    worker_task = asyncio.create_task(modsec_block_logger())
    yield
    worker_task.cancel()
    await http_client.aclose()


app = FastAPI(
    title="4thwall WAF API",
    description="Two-Tier Neural Web Application Firewall (DistilBERT Multi-Label CAPEC v2.1b)",
    version="3.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(dashboard_router)


@app.get("/health")
def health():
    """Returns model status and diagnostic metadata."""
    metrics = getattr(classifier, "metrics", {}) if classifier else {}
    return {
        "status": "ok",
        "model_loaded": classifier is not None,
        "model_path": getattr(classifier, "model_path", None) if classifier else None,
        "device": str(getattr(classifier, "device", "unknown")) if classifier else None,
        "version": metrics.get("model_version", "v2.1b"),
        "macro_f1": metrics.get("macro_f1", 0.9830),
        "exploit_catch_rate": metrics.get("exploit_catch_rate", 0.9941),
    }


@app.post("/score", response_model=ScoreResponse)
def score(request: ScoreRequest):
    """Score a single serialized request string."""
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        result = classifier.predict(request.request_text)
        logger.info(
            "score request -> %s (decision=%s, score=%.4f, firing=%s)",
            result["label"],
            result["decision"],
            result["score"],
            result["firing_labels"]
        )
        return ScoreResponse(**result)
    except Exception as e:
        logger.exception("score error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/score_batch", response_model=BatchScoreResponse)
def score_batch(request: BatchScoreRequest):
    """Score a batch of serialized request strings."""
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        results = classifier.predict_batch(request.requests)
        for i, r in enumerate(results):
            logger.info("score_batch[%d] -> %s (decision=%s, score=%.4f)", i, r["label"], r["decision"], r["score"])
        return BatchScoreResponse(results=[ScoreResponse(**r) for r in results])
    except Exception as e:
        logger.exception("score_batch error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/inspect", response_model=InspectResponse)
def inspect(request: InspectRequest):
    """Inspect structured HTTP components and return multi-label CAPEC analysis."""
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        serialized = serialize_request(
            method=request.method,
            path=request.path,
            ua=request.user_agent,
            ct=request.content_type,
            body=request.body
        )
        result = classifier.predict(serialized)
        return InspectResponse(**result)
    except Exception as e:
        logger.exception("inspect error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def reverse_proxy(request: Request, path: str):
    """Inline WAF inspection, blocking, and upstream proxying."""
    start_time = time.time()
    
    query_string = request.url.query
    user_agent = request.headers.get("user-agent", "")
    content_type = request.headers.get("content-type", "")
    method = request.method
    uri = "/" + path
    if query_string:
        uri += "?" + query_string
        
    start_time_iso = datetime.utcnow().isoformat() + "Z"
    
    # Read body before scoring to inspect payload attacks (SQLi, Code Injection, etc.)
    body = await request.body()
    body_text = body.decode("utf-8", errors="ignore") if body else ""
    
    request_text = serialize_request(
        method=method,
        path=uri,
        ua=user_agent,
        ct=content_type,
        body=body_text
    )
    
    ml_res = None
    ml_score = 0.0
    if classifier is not None:
        try:
            ml_res = classifier.predict(request_text)
            ml_score = ml_res.get("score", 0.0)
        except Exception as e:
            logger.error(f"Error scoring request: {e}")
            
    decision = make_decision(modsecurity_blocked=False, ml_result=ml_res, ml_score=ml_score)
    action = decision["action"]
    reason = decision["reason"]
    
    if action == "BLOCK":
        entry = {
            "timestamp": start_time_iso,
            "request_text": request_text,
            "modsecurity_blocked": False,
            "ml_score": ml_score,
            "action": action,
            "reason": reason,
            "blocked_by": "ml_model",
            "firing_labels": ml_res.get("firing_labels", []) if ml_res else [],
            "blocking_labels": ml_res.get("blocking_labels", []) if ml_res else [],
            "primary_attack": ml_res.get("primary_attack") if ml_res else None,
        }
        append_decision(entry)
        
        nginx_403 = (
            "<html>\r\n"
            "<head><title>403 Forbidden</title></head>\r\n"
            "<body>\r\n"
            "<center><h1>403 Forbidden</h1></center>\r\n"
            "<hr><center>4thwall WAF</center>\r\n"
            "</body>\r\n"
            "</html>\r\n"
        )
        return HTMLResponse(status_code=403, content=nginx_403)

    target_url = f"{DVWA_URL}/{path}"
    if query_string:
        target_url += "?" + query_string
         
    headers = dict(request.headers)
    headers.pop("host", None)
    
    try:
        proxy_resp = await http_client.request(
            method=request.method,
            url=target_url,
            headers=headers,
            content=body
        )
    except httpx.RequestError as exc:
        logger.error(f"Proxy error connecting to {exc.request.url}: {exc}")
        return JSONResponse(status_code=502, content={"error": "Bad Gateway"})
        
    entry = {
        "timestamp": start_time_iso,
        "request_text": request_text,
        "modsecurity_blocked": False,
        "ml_score": ml_score,
        "action": action,
        "reason": reason,
        "blocked_by": None,
        "firing_labels": ml_res.get("firing_labels", []) if ml_res else [],
        "alert_labels": ml_res.get("alert_labels", []) if ml_res else [],
        "primary_attack": ml_res.get("primary_attack") if ml_res else None,
    }
    append_decision(entry)
    
    response = Response(
        content=proxy_resp.content,
        status_code=proxy_resp.status_code,
    )
    for k, v in proxy_resp.headers.multi_items():
        if k.lower() not in ("content-encoding", "content-length"):
            response.headers.append(k, v)
    return response
