"""
Two-Tier Policy Engine for 4thwall_v3 WAF.
Enforces the calibrated Dual-Layer inspection policy:
- Layer 1: ModSecurity CRS (regex / protocol enforcement)
- Layer 2: Frozen Model v2.1b / HuggingFace WAF (semantic multi-label classification)
  - BLOCK Tier (10 classes): Code Injection, SQLi, OS Command Injection, Path Traversal, etc.
  - ALERT Tier (2 classes): Protocol Manipulation (272), Vulnerability Scanning (310)
  - ALLOW Tier: Benign / Normal traffic
"""

from typing import Any, Mapping
from ml.config import BLOCK_CLASSES, ALERT_CLASSES


def make_decision(
    modsecurity_blocked: bool = False,
    ml_score: float = 0.0,
    ml_result: Mapping[str, Any] | None = None
) -> dict:
    """
    Synthesize Layer 1 (ModSecurity CRS) and Layer 2 (ML Model) signals into an operational decision.

    Args:
        modsecurity_blocked: True if Nginx / ModSecurity CRS issued a 403 block.
        ml_score: Legacy scalar ML risk score (0.0 to 1.0).
        ml_result: Detailed prediction dictionary from WAFClassifier.predict().
            Expected keys: 'decision', 'blocking_labels', 'alert_labels', 'firing_labels', 'score'.

    Returns:
        dict: {
            'action': 'BLOCK' | 'ALERT' | 'ALLOW',
            'reason': str,
            'blocked_by': 'modsec' | 'ml_model' | None,
            'labels': list[str]
        }
    """
    # 1. First priority: Layer 1 ModSecurity block
    if modsecurity_blocked:
        return {
            "action": "BLOCK",
            "reason": "ModSecurity CRS rule triggered",
            "blocked_by": "modsec",
            "labels": []
        }

    # 2. Multi-label evaluation if detailed ml_result is available
    if ml_result is not None:
        blocking_labels = list(ml_result.get("blocking_labels") or [])
        alert_labels = list(ml_result.get("alert_labels") or [])
        firing_labels = list(ml_result.get("firing_labels") or [])

        # If blocking_labels not explicitly split, compute from firing_labels
        if not blocking_labels and not alert_labels and firing_labels:
            blocking_labels = [col for col in firing_labels if col in BLOCK_CLASSES]
            alert_labels = [col for col in firing_labels if col in ALERT_CLASSES]

        if blocking_labels:
            labels_str = ", ".join(blocking_labels)
            return {
                "action": "BLOCK",
                "reason": f"ML Block Tier triggered: {labels_str}",
                "blocked_by": "ml_model",
                "labels": blocking_labels
            }
        elif alert_labels:
            labels_str = ", ".join(alert_labels)
            return {
                "action": "ALERT",
                "reason": f"ML Alert Tier triggered: {labels_str}",
                "blocked_by": None,
                "labels": alert_labels
            }
        else:
            return {
                "action": "ALLOW",
                "reason": "Clean traffic / benign request",
                "blocked_by": None,
                "labels": []
            }

    # 3. Fallback to scalar ml_score for backward compatibility
    if ml_score > 0.9:
        return {
            "action": "BLOCK",
            "reason": f"High ML risk score ({ml_score:.2f})",
            "blocked_by": "ml_model",
            "labels": ["malicious"]
        }
    elif ml_score > 0.6:
        return {
            "action": "ALERT",
            "reason": f"Medium ML risk score ({ml_score:.2f})",
            "blocked_by": None,
            "labels": ["suspicious"]
        }
    else:
        return {
            "action": "ALLOW",
            "reason": "Low risk",
            "blocked_by": None,
            "labels": []
        }
