"""Explainable prototype capture-quality assessment for captured frames."""

from typing import Any, Dict, Iterable, Optional

import cv2
import numpy as np


# Initial prototype weights, not scientifically calibrated.
CAPTURE_QUALITY_WEIGHTS = {
    "visibility": 0.20,
    "sharpness": 0.15,
    "illumination": 0.15,
    "contrast": 0.10,
    "exposure": 0.15,
    "face_size": 0.10,
    "stability": 0.15,
}

MIN_VALID_FRAMES = 3
MIN_USABLE_FACE_FRAMES = 2
FACE_DETECTION_CONFIDENCE = 0.50
SHARPNESS_REFERENCE_VARIANCE = 100.0
MIN_FACE_AREA_RATIO = 0.02
GOOD_FACE_AREA_RATIO = 0.15
EXPOSURE_LOW_PIXEL = 16
EXPOSURE_HIGH_PIXEL = 239
EXPOSURE_TOLERANCE = 0.02
EXPOSURE_FAILURE_RATIO = 0.30
STABILITY_POSITION_TOLERANCE = 0.08
STABILITY_POSITION_SCALE = 0.15
STABILITY_AREA_TOLERANCE = 0.35
STABILITY_AREA_SCALE = 0.70

_DEFAULT_FACE_DETECTOR = None


def _bounded(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))


def _get_default_face_detector():
    """Use OpenCV's bundled Haar cascade if no project detector is supplied."""
    global _DEFAULT_FACE_DETECTOR
    if _DEFAULT_FACE_DETECTOR is None:
        cascade_path = (
            cv2.data.haarcascades
            + "haarcascade_frontalface_default.xml"
        )
        detector = cv2.CascadeClassifier(cascade_path)
        if detector.empty():
            raise RuntimeError(
                f"Could not load OpenCV face cascade: {cascade_path}"
            )
        _DEFAULT_FACE_DETECTOR = detector
    return _DEFAULT_FACE_DETECTOR


def _to_gray(frame):
    if not isinstance(frame, np.ndarray) or frame.size == 0:
        return None
    if frame.ndim == 2:
        gray = frame
    elif frame.ndim == 3 and frame.shape[2] == 1:
        gray = frame[:, :, 0]
    elif frame.ndim == 3 and frame.shape[2] == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif frame.ndim == 3 and frame.shape[2] == 4:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    else:
        return None
    if gray.size == 0 or not np.issubdtype(gray.dtype, np.number):
        return None
    if not np.isfinite(gray).all():
        return None
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    return gray


def _detect_faces(detector, frame, gray):
    if hasattr(detector, "get"):
        detections = detector.get(frame)
    elif hasattr(detector, "detectMultiScale"):
        detections = detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(20, 20),
        )
    elif callable(detector):
        detections = detector(frame)
    else:
        raise TypeError("face_detector must expose get, detectMultiScale, or be callable.")

    if detections is None:
        detections = []

    usable = []
    height, width = gray.shape[:2]
    for detection in detections:
        confidence = 1.0
        if isinstance(detection, dict):
            bbox = detection.get("bbox")
            confidence = detection.get("det_score", detection.get("confidence", 1.0))
        elif hasattr(detection, "bbox"):
            bbox = detection.bbox
            confidence = getattr(detection, "det_score", 1.0)
        else:
            values = np.asarray(detection).reshape(-1)
            if values.size < 4:
                continue
            x, y, box_width, box_height = values[:4]
            bbox = (x, y, x + box_width, y + box_height)

        try:
            confidence = float(confidence)
            coordinates = np.asarray(bbox, dtype=np.float64).reshape(-1)[:4]
        except (TypeError, ValueError):
            continue
        if (
            coordinates.size != 4
            or not np.isfinite(coordinates).all()
            or not np.isfinite(confidence)
            or confidence < FACE_DETECTION_CONFIDENCE
        ):
            continue

        x1, y1, x2, y2 = (int(round(value)) for value in coordinates)
        x1 = int(np.clip(x1, 0, width))
        x2 = int(np.clip(x2, 0, width))
        y1 = int(np.clip(y1, 0, height))
        y2 = int(np.clip(y2, 0, height))
        if x2 > x1 and y2 > y1:
            usable.append((x1, y1, x2, y2))
    return usable


def _frame_metrics(gray, bbox):
    x1, y1, x2, y2 = bbox
    face = gray[y1:y2, x1:x2]
    if face.size == 0:
        return None

    # Resizing makes the Laplacian reference less dependent on camera resolution.
    normalized_face = cv2.resize(face, (256, 256), interpolation=cv2.INTER_AREA)
    variance = float(cv2.Laplacian(normalized_face, cv2.CV_64F).var())
    sharpness = _bounded(
        variance / (variance + SHARPNESS_REFERENCE_VARIANCE)
    )

    mean_brightness = float(np.mean(face))
    illumination = _bounded(
        1.0 - abs(mean_brightness - 127.5) / 127.5
    )
    percentiles = np.percentile(face, (5, 95))
    contrast = _bounded(float(percentiles[1] - percentiles[0]) / 128.0)

    clipped_ratio = float(np.mean(
        (face <= EXPOSURE_LOW_PIXEL) | (face >= EXPOSURE_HIGH_PIXEL)
    ))
    exposure = _bounded(
        1.0
        - max(0.0, clipped_ratio - EXPOSURE_TOLERANCE)
        / (EXPOSURE_FAILURE_RATIO - EXPOSURE_TOLERANCE)
    )

    frame_area = float(gray.shape[0] * gray.shape[1])
    face_area_ratio = float((x2 - x1) * (y2 - y1)) / frame_area
    face_size = _bounded(
        (face_area_ratio - MIN_FACE_AREA_RATIO)
        / (GOOD_FACE_AREA_RATIO - MIN_FACE_AREA_RATIO)
    )
    center = (
        ((x1 + x2) / 2.0) / gray.shape[1],
        ((y1 + y2) / 2.0) / gray.shape[0],
    )
    return {
        "sharpness": sharpness,
        "illumination": illumination,
        "contrast": contrast,
        "exposure": exposure,
        "face_size": face_size,
        "face_area_ratio": max(face_area_ratio, 1e-8),
        "center": center,
    }


def _stability(frame_count, observations):
    visibility = len(observations) / float(frame_count) if frame_count else 0.0
    if len(observations) < 2:
        return _bounded(0.5 * visibility)

    position_scores = []
    area_scores = []
    for previous, current in zip(observations, observations[1:]):
        frame_gap = max(1, current["frame_index"] - previous["frame_index"])
        dx = current["center"][0] - previous["center"][0]
        dy = current["center"][1] - previous["center"][1]
        displacement = float(np.hypot(dx, dy)) / frame_gap
        excess_displacement = max(
            0.0,
            displacement - STABILITY_POSITION_TOLERANCE,
        )
        position_scores.append(
            float(np.exp(-excess_displacement / STABILITY_POSITION_SCALE))
        )

        area_change = abs(float(np.log(
            current["face_area_ratio"] / previous["face_area_ratio"]
        ))) / frame_gap
        excess_area_change = max(0.0, area_change - STABILITY_AREA_TOLERANCE)
        area_scores.append(
            float(np.exp(-excess_area_change / STABILITY_AREA_SCALE))
        )

    return _bounded(
        0.50 * visibility
        + 0.25 * float(np.mean(position_scores))
        + 0.25 * float(np.mean(area_scores))
    )


def assess_capture_quality(
    frames: Optional[Iterable[Any]],
    face_detector=None,
) -> Dict[str, Any]:
    """Return normalized frame-quality components and an overall prototype score.

    The default component weights are configurable prototype parameters; they
    have not been calibrated against a labelled capture-quality dataset.
    """
    if frames is None:
        frame_list = []
    elif isinstance(frames, np.ndarray) and frames.ndim in (2, 3):
        frame_list = [frames]
    else:
        try:
            frame_list = list(frames)
        except (TypeError, ValueError):
            frame_list = []
    frame_count = len(frame_list)
    component_names = tuple(CAPTURE_QUALITY_WEIGHTS)
    scores = {
        name: []
        for name in component_names
        if name not in {"visibility", "stability"}
    }
    observations = []
    valid_frames = 0
    detector_errors = 0

    if face_detector is None:
        try:
            face_detector = _get_default_face_detector()
        except (AttributeError, RuntimeError, cv2.error) as error:
            return {
                **{name: 0.0 for name in component_names},
                "overall_quality_score": 0.0,
                "valid": False,
                "frame_count": frame_count,
                "valid_frame_count": 0,
                "usable_face_frames": 0,
                "invalid_reason": f"face_detector_unavailable: {error}",
            }

    for frame_index, frame in enumerate(frame_list):
        try:
            gray = _to_gray(frame)
        except (TypeError, ValueError, cv2.error):
            gray = None
        if gray is None:
            continue
        valid_frames += 1

        try:
            faces = _detect_faces(face_detector, frame, gray)
        except Exception:
            # Detection failures are reported in the returned diagnostics and
            # must not abort the authentication attempt.
            detector_errors += 1
            continue
        if len(faces) != 1:
            continue

        try:
            metrics = _frame_metrics(gray, faces[0])
        except (TypeError, ValueError, FloatingPointError, cv2.error):
            detector_errors += 1
            continue
        if metrics is None:
            continue
        for name in scores:
            scores[name].append(metrics[name])
        observations.append({
            "frame_index": frame_index,
            "center": metrics["center"],
            "face_area_ratio": metrics["face_area_ratio"],
        })

    usable_face_frames = len(observations)
    visibility = (
        usable_face_frames / float(frame_count)
        if frame_count
        else 0.0
    )
    result = {
        name: _bounded(float(np.mean(values))) if values else 0.0
        for name, values in scores.items()
    }
    result["visibility"] = _bounded(visibility)
    result["stability"] = _stability(frame_count, observations)

    insufficient_frames = valid_frames < MIN_VALID_FRAMES
    insufficient_faces = usable_face_frames < MIN_USABLE_FACE_FRAMES
    valid = not insufficient_frames and not insufficient_faces
    if valid:
        quality_score = sum(
            CAPTURE_QUALITY_WEIGHTS[name] * result[name]
            for name in CAPTURE_QUALITY_WEIGHTS
        )
        invalid_reason = None
    else:
        quality_score = 0.0
        if insufficient_frames:
            invalid_reason = "insufficient_valid_frames"
        else:
            invalid_reason = "insufficient_usable_face_frames"
    if detector_errors and invalid_reason is None:
        invalid_reason = "some_face_detection_errors"

    result.update({
        "overall_quality_score": _bounded(quality_score),
        "valid": valid,
        "frame_count": frame_count,
        "valid_frame_count": valid_frames,
        "usable_face_frames": usable_face_frames,
        "detector_errors": detector_errors,
        "invalid_reason": invalid_reason,
    })
    return result
