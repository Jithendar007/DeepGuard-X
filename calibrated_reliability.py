"""Calibrated, explainable reliability estimation for DeepGuard-X.

This is deterministic prototype calibration, not a learned calibration model.
All parameters and component weights are centralized in ReliabilityConfig.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping

import numpy as np

from capture_quality import assess_capture_quality


@dataclass(frozen=True)
class ReliabilityConfig:
    face_weights: Mapping[str, float] = field(default_factory=lambda: {
        "score_confidence": 0.30,
        "sample_coverage": 0.20,
        "score_consistency": 0.30,
        "capture_quality": 0.20,
    })
    behaviour_weights: Mapping[str, float] = field(default_factory=lambda: {
        # Initial prototype weights; not scientifically calibrated.
        "valid_frame_ratio": 0.20,
        "feature_validity_ratio": 0.20,
        "temporal_consistency": 0.20,
        "behaviour_score": 0.25,
        "capture_quality": 0.15,
    })
    model_weights: Mapping[str, float] = field(default_factory=lambda: {
        "processed_frame_ratio": 0.25,
        "score_confidence": 0.25,
        "prediction_consistency": 0.30,
        "capture_quality": 0.20,
    })
    face_match_threshold: float = 0.45
    expected_face_samples: int = 3
    expected_heavy_frames: int = 20
    consistency_scale: float = 0.15
    behaviour_target_frames: int = 100
    behaviour_delta_scales: tuple = (0.02, 0.03, 2.0, 2.0, 2.0)
    calibration_power: float = 1.0


DEFAULT_CONFIG = ReliabilityConfig()


def _bounded(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))


def _weighted_quality(components, weights):
    valid = {
        name: _bounded(value)
        for name, value in components.items()
        if name in weights and np.isfinite(float(value))
    }
    total = sum(float(weights[name]) for name in valid)
    if not valid or total <= 0.0 or not np.isfinite(total):
        return 0.0
    return _bounded(
        sum(float(weights[name]) * valid[name] for name in valid) / total
    )


def calibrate_reliability(quality, config=DEFAULT_CONFIG):
    """Apply deterministic prototype calibration; no labels are inferred."""
    quality = _bounded(quality)
    return _bounded(quality ** float(config.calibration_power))


def _consistency(scores, scale):
    values = np.asarray(list(scores), dtype=np.float32)
    if values.size == 0 or not np.isfinite(values).all():
        return 0.0
    if values.size == 1:
        return 0.5
    return _bounded(np.exp(-float(np.std(values)) / float(scale)))


def _confidence(score):
    return _bounded(2.0 * abs(float(score) - 0.5))


def _valid_probability(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(value) and 0.0 <= value <= 1.0)


def _capture_quality_score(capture_quality_result):
    if not isinstance(capture_quality_result, dict):
        return 0.0, False
    score = capture_quality_result.get("overall_quality_score")
    if capture_quality_result.get("valid") is not True or not _valid_probability(score):
        return 0.0, False
    return float(score), True


def _valid_score_samples(scores):
    if not isinstance(scores, (list, tuple, np.ndarray)):
        return False
    try:
        values = np.asarray(scores, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        values.size > 0
        and np.isfinite(values).all()
        and np.logical_and(values >= 0.0, values <= 1.0).all()
    )


def _valid_model_evidence(score, diagnostics):
    if not _valid_probability(score) or not isinstance(diagnostics, dict):
        return False
    processed = diagnostics.get("processed_frames")
    skipped = diagnostics.get("skipped_frames")
    scores = diagnostics.get("scores")
    if (
        not isinstance(processed, int)
        or isinstance(processed, bool)
        or processed < 1
        or not isinstance(skipped, int)
        or isinstance(skipped, bool)
        or skipped < 0
        or not isinstance(scores, (list, tuple, np.ndarray))
    ):
        return False
    return len(scores) == processed and _valid_score_samples(scores)


def estimate_face_quality(
    face_score,
    face_scores: Iterable[float],
    captured_frames,
    config=DEFAULT_CONFIG,
    capture_quality_result=None,
):
    try:
        scores = list(face_scores) if face_scores is not None else []
    except TypeError:
        scores = []
    if capture_quality_result is None:
        capture_quality_result = assess_capture_quality(captured_frames)
    capture_quality, capture_quality_valid = _capture_quality_score(
        capture_quality_result
    )
    evidence_valid = (
        _valid_probability(face_score)
        and _valid_score_samples(scores)
    )
    if not evidence_valid:
        scores = []
    scores_array = np.asarray(scores, dtype=np.float32)
    components = {
        "score_confidence": _bounded(
            (float(face_score) - config.face_match_threshold)
            / (1.0 - config.face_match_threshold)
        ) if _valid_probability(face_score) else 0.0,
        "sample_coverage": _bounded(
            len(scores_array) / float(config.expected_face_samples)
        ),
        "score_consistency": _consistency(
            scores_array,
            config.consistency_scale,
        ),
        "capture_quality": capture_quality,
        "capture_quality_valid": capture_quality_valid,
        "evidence_valid": evidence_valid,
    }
    quality = _weighted_quality(components, config.face_weights)
    reliability = calibrate_reliability(quality, config) if evidence_valid else 0.0
    return components, reliability


def estimate_behaviour_quality(
    behaviour_score,
    behaviour_sequence,
    config=DEFAULT_CONFIG,
    capture_quality_result=None,
    captured_frames=None,
):
    if capture_quality_result is None:
        capture_quality_result = assess_capture_quality(captured_frames)
    capture_quality, capture_quality_valid = _capture_quality_score(
        capture_quality_result
    )
    try:
        sequence = list(behaviour_sequence) if behaviour_sequence is not None else []
    except TypeError:
        sequence = []
    required = (
        "ear", "mar", "yaw", "pitch", "roll",
        "d_ear", "d_mar", "d_yaw", "d_pitch", "d_roll",
    )
    valid_rows = []
    for row in sequence:
        try:
            values = np.asarray(
                [float(row[name]) for name in required],
                dtype=np.float64,
            )
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(values).all():
            valid_rows.append(values)

    feature_validity_ratio = (
        _bounded(len(valid_rows) / len(sequence))
        if sequence
        else 0.0
    )
    valid_frame_ratio = _bounded(
        len(valid_rows) / float(config.behaviour_target_frames)
    )
    if len(valid_rows) < 2:
        temporal_consistency = 0.0
    else:
        array = np.asarray(valid_rows, dtype=np.float32)
        delta_scales = np.asarray(config.behaviour_delta_scales)
        temporal_consistency = _bounded(
            1.0 / (
                1.0
                + float(np.mean(np.std(array[:, 5:], axis=0) / delta_scales))
            )
        )
    components = {
        "valid_frame_ratio": valid_frame_ratio,
        "feature_validity_ratio": feature_validity_ratio,
        "temporal_consistency": temporal_consistency,
        "behaviour_score": _bounded(behaviour_score)
        if _valid_probability(behaviour_score)
        else 0.0,
        "capture_quality": capture_quality,
        "capture_quality_valid": capture_quality_valid,
        "evidence_valid": bool(
            len(valid_rows) > 0 and _valid_probability(behaviour_score)
        ),
    }
    quality = _weighted_quality(components, config.behaviour_weights)
    evidence_valid = components["evidence_valid"]
    reliability = calibrate_reliability(quality, config) if evidence_valid else 0.0
    return components, reliability


def estimate_pad_quality(
    pad_score,
    diagnostics: Dict[str, Any],
    captured_frames,
    config=DEFAULT_CONFIG,
    capture_quality_result=None,
):
    return _estimate_model_quality(
        pad_score,
        diagnostics,
        captured_frames,
        config,
        capture_quality_result,
    )


def estimate_deepfake_quality(
    deepfake_score,
    diagnostics: Dict[str, Any],
    captured_frames,
    config=DEFAULT_CONFIG,
    capture_quality_result=None,
):
    return _estimate_model_quality(
        deepfake_score,
        diagnostics,
        captured_frames,
        config,
        capture_quality_result,
    )

def _estimate_model_quality(
    score,
    diagnostics,
    captured_frames,
    config,
    capture_quality_result=None,
):
    diagnostics = diagnostics or {}
    evidence_valid = _valid_model_evidence(score, diagnostics)
    scores = diagnostics.get("scores", [])
    processed = diagnostics.get("processed_frames", 0)
    skipped = diagnostics.get("skipped_frames", 0)
    if not isinstance(processed, int) or isinstance(processed, bool) or processed < 0:
        processed = 0
    if not isinstance(skipped, int) or isinstance(skipped, bool) or skipped < 0:
        skipped = 0
    total = processed + skipped
    if capture_quality_result is None:
        capture_quality_result = assess_capture_quality(captured_frames)
    capture_quality, capture_quality_valid = _capture_quality_score(
        capture_quality_result
    )
    if not evidence_valid:
        scores = []
        processed = 0
    components = {
        "processed_frame_ratio": _bounded(
            processed / float(config.expected_heavy_frames)
        ),
        "score_confidence": _confidence(score) if evidence_valid else 0.0,
        "prediction_consistency": _consistency(
            scores,
            config.consistency_scale,
        ),
        "capture_quality": capture_quality,
        "capture_quality_valid": capture_quality_valid,
        "evidence_valid": evidence_valid,
    }
    if total == 0:
        components["processed_frame_ratio"] = 0.0
    quality = _weighted_quality(components, config.model_weights)
    reliability = calibrate_reliability(quality, config) if evidence_valid else 0.0
    return components, reliability


def calculate_calibrated_reliability(
    face_score,
    behaviour_score,
    pad_score,
    deepfake_score,
    face_scores=None,
    behaviour_sequence=None,
    pad_diagnostics=None,
    deepfake_diagnostics=None,
    captured_frames=None,
    config=DEFAULT_CONFIG,
    capture_quality_result=None,
):
    if capture_quality_result is None:
        capture_quality_result = assess_capture_quality(captured_frames)

    face_components, face_reliability = estimate_face_quality(
        face_score, face_scores, captured_frames, config, capture_quality_result
    )
    behaviour_components, behaviour_reliability = estimate_behaviour_quality(
        behaviour_score,
        behaviour_sequence,
        config,
        capture_quality_result,
        captured_frames,
    )
    pad_components, pad_reliability = estimate_pad_quality(
        pad_score, pad_diagnostics, captured_frames, config, capture_quality_result
    )
    deepfake_components, deepfake_reliability = estimate_deepfake_quality(
        deepfake_score, deepfake_diagnostics, captured_frames, config,
        capture_quality_result
    )
    return {
        "components": {
            "face": face_components,
            "behaviour": behaviour_components,
            "pad": pad_components,
            "deepfake": deepfake_components,
        },
        "reliabilities": {
            "face": face_reliability,
            "behaviour": behaviour_reliability,
            "pad": pad_reliability,
            "deepfake": deepfake_reliability,
        },
        "config": config,
        "capture_quality": capture_quality_result,
    }
