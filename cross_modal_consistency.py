"""Explainable cross-modal contradiction analysis for DeepGuard-X."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import numpy as np


class EvidenceState(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class AttackSeverity(str, Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    HIGH_RISK = "HIGH_RISK"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class EvidenceThresholds:
    face_high: float = 0.75
    face_medium: float = 0.55
    behaviour_high: float = 0.75
    behaviour_medium: float = 0.55
    pad_high: float = 0.75
    pad_medium: float = 0.50
    deepfake_high: float = 0.75
    deepfake_medium: float = 0.55


DEFAULT_THRESHOLDS = EvidenceThresholds()


@dataclass(frozen=True)
class CrossModalResult:
    face_state: str
    behaviour_state: str
    pad_state: str
    deepfake_state: str
    consistency_score: float
    detected_conflicts: List[str]
    attack_severity: str
    attack_type: str
    security_override: bool
    final_security_decision: str
    invalid_input: bool = False

    def as_dict(self):
        return {
            "face_state": self.face_state,
            "behaviour_state": self.behaviour_state,
            "pad_state": self.pad_state,
            "deepfake_state": self.deepfake_state,
            "consistency_score": self.consistency_score,
            "detected_conflicts": self.detected_conflicts,
            "attack_severity": self.attack_severity,
            "attack_type": self.attack_type,
            "security_override": self.security_override,
            "final_security_decision": self.final_security_decision,
            "invalid_input": self.invalid_input,
        }


def _invalid_result():
    return CrossModalResult(
        face_state=EvidenceState.LOW.value,
        behaviour_state=EvidenceState.LOW.value,
        pad_state=EvidenceState.LOW.value,
        deepfake_state=EvidenceState.LOW.value,
        consistency_score=0.0,
        detected_conflicts=["INVALID_OR_MISSING_EVIDENCE"],
        attack_severity=AttackSeverity.CRITICAL.value,
        attack_type="INVALID_OR_MISSING_EVIDENCE",
        security_override=True,
        final_security_decision="REJECT",
        invalid_input=True,
    )


def _finite_scores(scores):
    values = np.asarray(list(scores), dtype=np.float64)
    return values.size == 4 and np.isfinite(values).all()


def _state(score, high, medium):
    if score >= high:
        return EvidenceState.HIGH.value
    if score >= medium:
        return EvidenceState.MEDIUM.value
    return EvidenceState.LOW.value


def _pairwise_agreement(first, second):
    return float(np.clip(1.0 - abs(first - second), 0.0, 1.0))


def _consistency_score(scores):
    values = list(scores.values())
    agreements = [
        _pairwise_agreement(values[0], values[1]),
        _pairwise_agreement(values[0], values[2]),
        _pairwise_agreement(values[0], values[3]),
        _pairwise_agreement(values[1], values[2]),
        _pairwise_agreement(values[1], values[3]),
        _pairwise_agreement(values[2], values[3]),
    ]
    return float(np.mean(agreements))


def analyze_cross_modal_consistency(
    face_score,
    behaviour_score,
    pad_score,
    deepfake_score,
    face_match=True,
    thresholds=DEFAULT_THRESHOLDS,
):
    scores = {
        "face": face_score,
        "behaviour": behaviour_score,
        "pad": pad_score,
        "deepfake": deepfake_score,
    }
    if not _finite_scores(scores.values()):
        return _invalid_result()
    if not face_match:
        return CrossModalResult(
            face_state=_state(face_score, thresholds.face_high, thresholds.face_medium),
            behaviour_state=_state(behaviour_score, thresholds.behaviour_high, thresholds.behaviour_medium),
            pad_state=_state(pad_score, thresholds.pad_high, thresholds.pad_medium),
            deepfake_state=_state(deepfake_score, thresholds.deepfake_high, thresholds.deepfake_medium),
            consistency_score=_consistency_score(scores),
            detected_conflicts=["FACE_ID_NO_VALID_MATCH"],
            attack_severity=AttackSeverity.HIGH_RISK.value,
            attack_type="FACE_ID_NO_VALID_MATCH",
            security_override=True,
            final_security_decision="REJECT",
        )

    states = {
        "face": _state(face_score, thresholds.face_high, thresholds.face_medium),
        "behaviour": _state(behaviour_score, thresholds.behaviour_high, thresholds.behaviour_medium),
        "pad": _state(pad_score, thresholds.pad_high, thresholds.pad_medium),
        "deepfake": _state(deepfake_score, thresholds.deepfake_high, thresholds.deepfake_medium),
    }
    if states["face"] == EvidenceState.LOW.value:
        return CrossModalResult(
            face_state=states["face"],
            behaviour_state=states["behaviour"],
            pad_state=states["pad"],
            deepfake_state=states["deepfake"],
            consistency_score=_consistency_score(scores),
            detected_conflicts=["FACE_ID_LOW"],
            attack_severity=AttackSeverity.HIGH_RISK.value,
            attack_type="FACE_ID_LOW",
            security_override=True,
            final_security_decision="REJECT",
        )
    conflicts = []
    strong_conflicts = []

    if states["face"] in (EvidenceState.HIGH.value, EvidenceState.MEDIUM.value) and states["pad"] == EvidenceState.LOW.value:
        conflicts.append("IDENTITY_LIVENESS_CONFLICT")
        strong_conflicts.append("IDENTITY_LIVENESS_CONFLICT")
    if states["face"] == EvidenceState.HIGH.value and states["deepfake"] == EvidenceState.LOW.value:
        conflicts.append("IDENTITY_DEEPFAKE_CONFLICT")
        strong_conflicts.append("IDENTITY_DEEPFAKE_CONFLICT")
    if states["face"] == EvidenceState.HIGH.value and states["behaviour"] == EvidenceState.LOW.value:
        conflicts.append("IDENTITY_BEHAVIOUR_CONFLICT")

    if len(strong_conflicts) >= 2 or (
        strong_conflicts
        and states["behaviour"] == EvidenceState.LOW.value
    ):
        severity = AttackSeverity.CRITICAL.value
    elif strong_conflicts:
        severity = AttackSeverity.HIGH_RISK.value
    elif conflicts:
        severity = AttackSeverity.WARNING.value
    else:
        severity = AttackSeverity.NORMAL.value

    if severity == AttackSeverity.CRITICAL.value:
        attack_type = "MULTI_MODAL_CONFLICT"
    elif conflicts:
        attack_type = conflicts[0]
    else:
        attack_type = "NONE"

    pad_veto = float(pad_score) < thresholds.pad_medium
    security_override = (
        pad_veto
        or severity in (
            AttackSeverity.HIGH_RISK.value,
            AttackSeverity.CRITICAL.value,
        )
    )
    final_security_decision = "REJECT" if security_override else "ACCEPT"
    return CrossModalResult(
        face_state=states["face"],
        behaviour_state=states["behaviour"],
        pad_state=states["pad"],
        deepfake_state=states["deepfake"],
        consistency_score=_consistency_score(scores),
        detected_conflicts=conflicts,
        attack_severity=severity,
        attack_type=attack_type,
        security_override=security_override,
        final_security_decision=final_security_decision,
    )
