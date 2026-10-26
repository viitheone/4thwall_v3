"""
Pydantic schemas for 4thwall_v3 WAF API.
Maintains full backward compatibility while providing rich multi-label CAPEC diagnostics.
"""

from typing import List, Dict, Optional
from pydantic import BaseModel, Field


class ScoreRequest(BaseModel):
    request_text: str = Field(..., description="Serialized request string formatted with markup tokens or raw HTTP")


class ScoreResponse(BaseModel):
    score: float = Field(..., description="Highest attack probability score (0.0 - 1.0)")
    label: str = Field(..., description="Classification verdict: 'benign' or 'malicious'")
    confidence: float = Field(..., description="Confidence score for the predicted verdict")
    decision: Optional[str] = Field("ALLOW", description="Enforcement action: ALLOW, ALERT, or BLOCK")
    is_attack: Optional[bool] = Field(False, description="True if any attack class exceeded threshold")
    firing_labels: Optional[List[str]] = Field(default_factory=list, description="All CAPEC classes exceeding calibrated thresholds")
    blocking_labels: Optional[List[str]] = Field(default_factory=list, description="Classes triggering automated inline blocking")
    alert_labels: Optional[List[str]] = Field(default_factory=list, description="Classes triggering passive alerting")
    primary_attack: Optional[str] = Field(None, description="Primary attack category detected")
    probabilities: Optional[Dict[str, float]] = Field(None, description="Per-class probabilities across all 13 CAPEC categories")


class BatchScoreRequest(BaseModel):
    requests: List[str] = Field(..., description="List of serialized request strings")


class BatchScoreResponse(BaseModel):
    results: List[ScoreResponse] = Field(..., description="List of score results")


class InspectRequest(BaseModel):
    method: str = Field("GET", description="HTTP Method (GET, POST, etc.)")
    path: str = Field("/", description="Request URI path with query parameters")
    user_agent: Optional[str] = Field("", description="Client User-Agent header")
    content_type: Optional[str] = Field("", description="Content-Type header")
    body: Optional[str] = Field("", description="Raw request body/payload")


class InspectResponse(ScoreResponse):
    serialized_text: Optional[str] = Field(None, description="Canonical serialized representation scored by DistilBERT")
