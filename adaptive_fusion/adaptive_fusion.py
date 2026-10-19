import json
import os
from datetime import datetime, timezone

import cv2
import numpy as np


BASELINE_WEIGHTS = {
    "face": 0.25,
    "behaviour": 0.25,
    "pad": 0.25,
    "deepfake": 0.25,
}
V2_BASELINE_WEIGHTS = {
    "face": 0.30,
    "behaviour": 0.25,
    "pad": 0.25,
    "deepfake": 0.20,
}
ADAPTIVE_THRESHOLD = 0.70
PAD_SECURITY_VETO_THRESHOLD = 0.50
LOG_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "adaptive_fusion_log.jsonl"
)


def _clamp(value, lower=0.0, upper=1.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, lower, upper))


def _capture_quality(frames):
    if not frames:
        return 0.0

    quality_values = []
    for frame in frames:
        if frame is None or getattr(frame, "size", 0) == 0:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharpness = _clamp(cv2.Laplacian(gray, cv2.CV_64F).var() / 500.0)
        brightness = float(np.mean(gray))
        illumination = _clamp(1.0 - abs(brightness - 128.0) / 128.0)
        quality_values.append(0.60 * sharpness + 0.40 * illumination)

    return _clamp(np.mean(quality_values)) if quality_values else 0.0


def _score_consistency(scores):
    values = np.asarray(scores, dtype=np.float32)
    if values.size and not np.isfinite(values).all():
        return 0.0
    if values.size < 2:
        return 0.0 if values.size == 0 else 0.5
    return _clamp(np.exp(-float(np.std(values)) / 0.15))


def _coverage(processed, skipped):
    total = int(processed) + int(skipped)
    return _clamp(processed / total) if total else 0.0


def estimate_face_reliability(face_score, face_scores, capture_quality):
    scores = np.asarray(face_scores, dtype=np.float32)
    score_quality = _clamp((face_score - 0.45) / 0.55)
    sample_coverage = _clamp(len(scores) / 3.0)
    consistency = _score_consistency(scores)
    return _clamp(
        0.35 * score_quality
        + 0.25 * sample_coverage
        + 0.25 * consistency
        + 0.15 * capture_quality
    )


def estimate_behaviour_reliability(behaviour_score, behaviour_sequence):
    if not behaviour_sequence:
        return 0.0

    values = []
    for row in behaviour_sequence:
        try:
            values.append([
                float(row["ear"]),
                float(row["mar"]),
                float(row["d_ear"]),
                float(row["d_mar"]),
            ])
        except (KeyError, TypeError, ValueError):
            continue

    if not values:
        return 0.0

    array = np.asarray(values, dtype=np.float32)
    finite_ratio = _clamp(np.isfinite(array).all(axis=1).mean())
    frame_coverage = _clamp(len(array) / 100.0)
    deltas = array[:, 2:]
    delta_scales = np.array([0.02, 0.03], dtype=np.float32)
    temporal_consistency = _clamp(
        1.0 / (1.0 + float(np.mean(np.std(deltas, axis=0) / delta_scales)))
    )
    return _clamp(
        0.35 * frame_coverage
        + 0.25 * finite_ratio
        + 0.20 * temporal_consistency
        + 0.20 * _clamp(behaviour_score)
    )


def estimate_model_reliability(diagnostics, score, capture_quality):
    diagnostics = diagnostics or {}
    processed = diagnostics.get("processed_frames", 0)
    skipped = diagnostics.get("skipped_frames", 0)
    scores = diagnostics.get("scores", [])
    return _clamp(
        0.45 * _coverage(processed, skipped)
        + 0.35 * _score_consistency(scores)
        + 0.10 * _clamp(2.0 * abs(float(score) - 0.5))
        + 0.10 * capture_quality
    )


def estimate_pad_reliability(pad_score, diagnostics, capture_quality):
    return estimate_model_reliability(
        diagnostics,
        pad_score,
        capture_quality,
    )


def estimate_deepfake_reliability(
    deepfake_score,
    diagnostics,
    capture_quality,
):
    return estimate_model_reliability(
        diagnostics,
        deepfake_score,
        capture_quality,
    )


def _finite_scalar(value):
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _valid_diagnostics(diagnostics):
    if not isinstance(diagnostics, dict):
        return False
    processed = diagnostics.get("processed_frames")
    skipped = diagnostics.get("skipped_frames")
    scores = diagnostics.get("scores")
    if not isinstance(processed, int) or not isinstance(skipped, int):
        return False
    if processed < 1 or skipped < 0 or not isinstance(scores, (list, tuple)):
        return False
    values = np.asarray(scores, dtype=np.float32)
    return (
        len(values) == processed
        and len(values) > 0
        and np.isfinite(values).all()
    )


def _valid_behaviour_sequence(sequence):
    if not isinstance(sequence, (list, tuple)) or not sequence:
        return False
    required = ("ear", "mar", "d_ear", "d_mar")
    for row in sequence:
        if not isinstance(row, dict):
            return False
        if not all(key in row and _finite_scalar(row[key]) for key in required):
            return False
    return True


def calculate_adaptive_weights(
    reliabilities,
    baseline_weights=BASELINE_WEIGHTS,
):
    if (
        not isinstance(reliabilities, dict)
        or set(reliabilities) != set(baseline_weights)
        or not all(_finite_scalar(value) for value in reliabilities.values())
    ):
        return None

    effective_weights = {
        name: baseline_weights[name] * _clamp(reliabilities[name])
        for name in baseline_weights
    }
    weight_total = sum(effective_weights.values())
    if not np.isfinite(weight_total) or weight_total <= 0.0:
        return None

    return {
        name: value / weight_total
        for name, value in effective_weights.items()
    }


def calculate_adaptive_fusion(
    face_score,
    behaviour_score,
    pad_score,
    deepfake_score,
    face_scores=None,
    behaviour_sequence=None,
    pad_diagnostics=None,
    deepfake_diagnostics=None,
    captured_frames=None,
    face_match=False,
    behaviour_available=True,
    pad_available=True,
    deepfake_available=True,
):
    scores = (face_score, behaviour_score, pad_score, deepfake_score)
    if (
        not all(_finite_scalar(score) for score in scores)
        or not face_match
        or not behaviour_available
        or not pad_available
        or not deepfake_available
        or not _valid_behaviour_sequence(behaviour_sequence)
        or not _valid_diagnostics(pad_diagnostics)
        or not _valid_diagnostics(deepfake_diagnostics)
    ):
        return {
            "capture_quality": 0.0,
            "reliabilities": {name: 0.0 for name in V2_BASELINE_WEIGHTS},
            "adaptive_weights": {name: 0.0 for name in V2_BASELINE_WEIGHTS},
            "adaptive_score": 0.0,
            "pad_veto": _finite_scalar(pad_score) and float(pad_score) < PAD_SECURITY_VETO_THRESHOLD,
            "adaptive_decision": "REJECT",
            "unavailable_reason": "Adaptive fusion unavailable: invalid or missing evidence.",
        }

    capture_quality = _capture_quality(captured_frames or [])
    reliabilities = {
        "face": estimate_face_reliability(
            face_score,
            face_scores or [],
            capture_quality,
        ),
        "behaviour": estimate_behaviour_reliability(
            behaviour_score,
            behaviour_sequence or [],
        ),
        "pad": estimate_pad_reliability(
            pad_score,
            pad_diagnostics,
            capture_quality,
        ),
        "deepfake": estimate_deepfake_reliability(
            deepfake_score,
            deepfake_diagnostics,
            capture_quality,
        ),
    }

    adaptive_weights = calculate_adaptive_weights(
        reliabilities,
        baseline_weights=V2_BASELINE_WEIGHTS,
    )
    if adaptive_weights is None:
        return {
            "capture_quality": capture_quality,
            "reliabilities": reliabilities,
            "adaptive_weights": {name: 0.0 for name in V2_BASELINE_WEIGHTS},
            "adaptive_score": 0.0,
            "pad_veto": float(pad_score) < PAD_SECURITY_VETO_THRESHOLD,
            "adaptive_decision": "REJECT",
            "unavailable_reason": "Adaptive fusion unavailable: no reliable evidence.",
        }

    adaptive_score = (
        adaptive_weights["face"] * face_score
        + adaptive_weights["behaviour"] * behaviour_score
        + adaptive_weights["pad"] * pad_score
        + adaptive_weights["deepfake"] * deepfake_score
    )

    # A PAD score below the existing 0.50 PAD boundary is treated as
    # presentation-attack evidence and cannot be overridden by other models.
    pad_veto = float(pad_score) < PAD_SECURITY_VETO_THRESHOLD
    adaptive_decision = (
        "REJECT"
        if pad_veto or adaptive_score < ADAPTIVE_THRESHOLD
        else "ACCEPT"
    )

    return {
        "capture_quality": capture_quality,
        "reliabilities": reliabilities,
        "adaptive_weights": adaptive_weights,
        "adaptive_score": float(adaptive_score),
        "pad_veto": pad_veto,
        "adaptive_decision": adaptive_decision,
        "unavailable_reason": None,
    }


def log_fusion_attempt(record):
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **record,
    }
    with open(LOG_FILE, "a", encoding="utf-8") as file:
        file.write(json.dumps(payload, sort_keys=True) + "\n")
