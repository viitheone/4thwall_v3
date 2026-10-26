"""
Comprehensive integration test suite for 4thwall_v3 WAF (Frozen Model v2.1b).
Verifies serializer protocol, two-tier policy, model inference, and API endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from ml.config import (
    LABEL_COLS,
    BLOCK_CLASSES,
    ALERT_CLASSES,
    CALIBRATED_THRESHOLDS,
    DEFAULT_UA,
    HF_MODEL_REPO
)
from ml.preprocess import serialize_request, parse_serialized_request
from policy.decision import make_decision
from ml.infer import WAFClassifier
from api.main import app


def test_serializer_protocol():
    """Verify serialization adheres to canonical protocol."""
    # 1. User-Agent fallback
    s = serialize_request(method="GET", path="/test", ua="")
    assert f"[UA] {DEFAULT_UA}" in s

    # 2. Method normalization
    s = serialize_request(method="post", path="login")
    assert s.startswith("[METHOD] POST")
    assert "[PATH] /login" in s

    # 3. Body truncation to 500 chars
    long_body = "A" * 800
    s = serialize_request(method="POST", path="/api", body=long_body)
    assert f"[BODY] {'A' * 500}" in s
    assert f"[BODY] {'A' * 501}" not in s

    # 4. Roundtrip parsing
    parsed = parse_serialized_request(s)
    assert parsed["method"] == "POST"
    assert parsed["path"] == "/api"
    assert len(parsed["body"]) == 500


def test_two_tier_policy():
    """Verify two-tier operational enforcement matrix."""
    # ModSec priority
    d = make_decision(modsecurity_blocked=True)
    assert d["action"] == "BLOCK"
    assert d["blocked_by"] == "modsec"

    # BLOCK tier classes
    for b_class in BLOCK_CLASSES:
        res = {
            "blocking_labels": [b_class],
            "alert_labels": [],
            "firing_labels": [b_class],
            "score": 0.95
        }
        d = make_decision(modsecurity_blocked=False, ml_result=res)
        assert d["action"] == "BLOCK", f"Failed to BLOCK {b_class}"
        assert d["blocked_by"] == "ml_model"

    # ALERT tier classes
    for a_class in ALERT_CLASSES:
        res = {
            "blocking_labels": [],
            "alert_labels": [a_class],
            "firing_labels": [a_class],
            "score": 0.85
        }
        d = make_decision(modsecurity_blocked=False, ml_result=res)
        assert d["action"] == "ALERT", f"Failed to ALERT {a_class}"
        assert d["blocked_by"] is None

    # Benign / clean traffic
    res_clean = {
        "blocking_labels": [],
        "alert_labels": [],
        "firing_labels": ["000 - Normal"],
        "score": 0.01
    }
    d = make_decision(modsecurity_blocked=False, ml_result=res_clean)
    assert d["action"] == "ALLOW"
    assert d["blocked_by"] is None


def test_model_inference_local():
    """Verify local model loading and scoring on critical vectors."""
    classifier = WAFClassifier(model_path="models/huggingface_model")

    # 1. Benign request
    res_clean = classifier.score_request(
        method="GET",
        path="/blog/index.php/2020/04/04/hello-world",
        ua=DEFAULT_UA
    )
    assert res_clean["decision"] == "ALLOW"
    assert "000 - Normal" in res_clean["firing_labels"]

    # 2. SQL Injection
    res_sqli = classifier.score_request(
        method="GET",
        path="/?id=1%27%20OR%201=1--",
        ua=DEFAULT_UA
    )
    assert res_sqli["decision"] == "BLOCK"
    assert "66 - SQL Injection" in res_sqli["blocking_labels"]

    # 3. Path Traversal
    res_lfi = classifier.score_request(
        method="GET",
        path="/../../../../etc/passwd",
        ua=DEFAULT_UA
    )
    assert res_lfi["decision"] == "BLOCK"
    assert "126 - Path Traversal" in res_lfi["blocking_labels"]

    # 4. HTTP Verb Tampering
    res_verb = classifier.score_request(
        method="TRACK",
        path="/",
        ua=DEFAULT_UA
    )
    assert res_verb["decision"] == "BLOCK"
    assert "274 - HTTP Verb Tampering" in res_verb["blocking_labels"]


def test_model_inference_huggingface_hub():
    """Verify loading and scoring directly from HuggingFace Hub repo."""
    classifier = WAFClassifier(model_path=HF_MODEL_REPO)

    # SQL Injection in POST body
    res = classifier.score_request(
        method="POST",
        path="/login.php",
        ua=DEFAULT_UA,
        ct="application/x-www-form-urlencoded",
        body="username=admin%27+OR+1%3D1--"
    )
    assert res["decision"] == "BLOCK"
    assert "66 - SQL Injection" in res["blocking_labels"]
    assert res["score"] > 0.90



def test_fastapi_endpoints():
    """Verify FastAPI API endpoints with TestClient."""
    # Ensure classifier is initialized on app
    classifier = WAFClassifier(model_path="models/huggingface_model")
    app.state.waf_classifier = classifier
    import api.main as main_mod
    main_mod.classifier = classifier

    with TestClient(app) as client:
        # /health
        h_resp = client.get("/health")
        assert h_resp.status_code == 200
        h_data = h_resp.json()
        assert h_data["model_loaded"] is True
        assert h_data["version"] == "v2.1b"

        # /inspect endpoint
        i_resp = client.post("/inspect", json={
            "method": "GET",
            "path": "/?id=1' OR '1'='1",
            "user_agent": DEFAULT_UA
        })
        assert i_resp.status_code == 200
        i_data = i_resp.json()
        assert i_data["decision"] == "BLOCK"
        assert "66 - SQL Injection" in i_data["blocking_labels"]

        # /score endpoint
        s_resp = client.post("/score", json={
            "request_text": "[METHOD] GET [PATH] /blog [UA] Mozilla/5.0"
        })
        assert s_resp.status_code == 200
        s_data = s_resp.json()
        assert "label" in s_data
        assert "decision" in s_data

        # /dashboard/ai-status
        ai_resp = client.get("/dashboard/ai-status")
        assert ai_resp.status_code == 200
        ai_data = ai_resp.json()
        assert "DistilBERT" in ai_data["architecture"]
        assert ai_data["f1"] >= 0.98

        # /dashboard/summary
        sum_resp = client.get("/dashboard/summary")
        assert sum_resp.status_code == 200


if __name__ == "__main__":
    pytest.main(["-v", __file__])
