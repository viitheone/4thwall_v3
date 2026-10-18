import logging
import os

import json
import time
import asyncio
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx

from api.dashboard import router as dashboard_router
from api.schemas import BatchScoreRequest, BatchScoreResponse, ScoreRequest, ScoreResponse
from ml.infer import WAFClassifier
from ml.preprocess import serialize_request
from policy.decision import make_decision

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="4thwall WAF API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

classifier: WAFClassifier | None = None

app.include_router(dashboard_router)

RETRAIN_LOG_PATH = os.environ.get("RETRAIN_LOG", "logs/retrain_log.json")
ACCESS_LOG_PATH = os.environ.get("ACCESS_LOG_PATH", "logs/access.log")
OFFSET_PATH = os.environ.get("OFFSET_PATH", "logs/.waf_log_worker_offset")

async def modsec_block_logger():
    """Background task: tail Nginx access.log for ModSec 403 blocks.

    1. Writes every ModSec 403 to the *decision log* so the dashboard shows it.
    2. If the ML model disagrees (score < 0.6), also writes to the *retrain log*
       so we can fine-tune the model on ModSec's judgement.
    """
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
                    
                    # Log every ModSec 403 to the decision log for the dashboard
                    if status == 403 and data.get("request_time", 1.0) < 0.1:
                        query = data.get("query", "")
                        uri = data.get("path", "/")
                        if query: uri += "?" + query
                        
                        row = {
                            "method": data.get("method", "GET"),
                            "path": uri,
                            "status": 403,
                            "user_agent": data.get("user_agent", ""),
                            "request_time": 0.0
                        }
                        req_text = serialize_request(row)
                        
                        # Run ML on the blocked request for scoring / retrain analysis
                        ml_score = 0.0
                        if classifier is not None:
                            try:
                                res = classifier.predict(req_text)
                                ml_score = res["score"]
                            except Exception:
                                pass
                        
                        # Always log to decision log so dashboard shows the block
                        append_decision({
                            "timestamp": data.get("time", ""),
                            "request_text": req_text,
                            "modsecurity_blocked": True,
                            "ml_score": ml_score,
                            "action": "BLOCK",
                            "reason": "ModSecurity rule triggered",
                            "blocked_by": "modsec",
                        })
                        
                        # If ML disagrees, also log to retrain file
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
        except Exception as e:
            logger.error(f"Error in modsec block logger: {e}")


@app.on_event("startup")
async def startup():
    global classifier
    model_path = os.environ.get("MODEL_PATH", "viitheone/waf-model")
    try:
        classifier = WAFClassifier(model_path)
        logger.info("WAFClassifier loaded from %s", model_path)
    except Exception as e:
        classifier = None
        logger.warning("Failed to load model from %s: %s  — /score endpoints will return 503", model_path, e)
    app.state.waf_classifier = classifier
    
    asyncio.create_task(modsec_block_logger())


@app.get("/health")
def health():
    """Returns model status."""
    return {
        "status": "ok",
        "model_loaded": classifier is not None,
    }


@app.post("/score", response_model=ScoreResponse)
def score(request: ScoreRequest):
    """Score a single serialized request."""
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        result = classifier.predict(request.request_text)
        logger.info("score request -> %s (score=%.4f)", result["label"], result["score"])
        return ScoreResponse(**result)
    except Exception as e:
        logger.exception("score error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/score_batch", response_model=BatchScoreResponse)
def score_batch(request: BatchScoreRequest):
    """Score a batch of serialized requests."""
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        results = classifier.predict_batch(request.requests)
        for i, r in enumerate(results):
            logger.info("score_batch[%d] -> %s (score=%.4f)", i, r["label"], r["score"])
        return BatchScoreResponse(results=[ScoreResponse(**r) for r in results])
    except Exception as e:
        logger.exception("score_batch error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))



DVWA_URL = os.environ.get("DVWA_URL", "http://dvwa:80")
DECISION_LOG_PATH = os.environ.get("DECISION_LOG", "logs/decision_log.json")

def append_decision(entry: dict) -> None:
    os.makedirs(os.path.dirname(DECISION_LOG_PATH), exist_ok=True)
    with open(DECISION_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

http_client = httpx.AsyncClient(timeout=30.0)

@app.on_event("shutdown")
async def shutdown():
    await http_client.aclose()


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def reverse_proxy(request: Request, path: str):
    """Inline WAF blocking and proxying."""
    start_time = time.time()
    
    query_string = request.url.query
    user_agent = request.headers.get("user-agent", "")
    method = request.method
    uri = "/" + path
    if query_string:
        uri += "?" + query_string
        
    start_time_iso = datetime.utcnow().isoformat() + "Z"
    
    row = {
        "method": method,
        "path": uri,
        "query": query_string,
        "status": 200, 
        "user_agent": user_agent,
        "request_time": 0.0,
    }
    
    request_text = serialize_request(row)
    
    ml_score = 0.0
    if classifier is not None:
        try:
            result = classifier.predict(request_text)
            ml_score = result["score"]
        except Exception as e:
            logger.error(f"Error scoring request: {e}")
            
    decision = make_decision(modsecurity_blocked=False, ml_score=ml_score)
    action = decision["action"]
    reason = decision["reason"]
    
    if action == "BLOCK":
        row["status"] = 403
        row["request_time"] = time.time() - start_time
        entry = {
            "timestamp": start_time_iso,
            "request_text": serialize_request(row),
            "modsecurity_blocked": False,
            "ml_score": ml_score,
            "action": action,
            "reason": reason,
            "blocked_by": "ml_model",
        }
        append_decision(entry)
        
        nginx_403 = (
            "<html>\r\n"
            "<head><title>403 Forbidden</title></head>\r\n"
            "<body>\r\n"
            "<center><h1>403 Forbidden</h1></center>\r\n"
            "<hr><center>nginx</center>\r\n"
            "</body>\r\n"
            "</html>\r\n"
        )
        return HTMLResponse(status_code=403, content=nginx_403)

    target_url = f"{DVWA_URL}/{path}"
    if query_string:
         target_url += "?" + query_string
         
    body = await request.body()
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
        
    row["status"] = proxy_resp.status_code
    row["request_time"] = time.time() - start_time
    
    entry = {
        "timestamp": start_time_iso,
        "request_text": serialize_request(row),
        "modsecurity_blocked": False,
        "ml_score": ml_score,
        "action": action,
        "reason": reason,
        "blocked_by": None,
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
