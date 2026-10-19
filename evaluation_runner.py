"""Structured evaluation records for DeepGuard-X authentication attempts."""

import csv
import json
import os
import time
import uuid
from datetime import datetime, timezone


EVALUATION_CSV = "evaluation_results.csv"
EVALUATION_JSONL = "evaluation_results.jsonl"
LINKED_EVALUATION_CSV = "evaluation_results_linked.csv"
SCENARIOS = {
    "GENUINE",
    "IMPOSTOR",
    "DEGRADED_CAPTURE",
    "FAILURE_CASE",
    "UNIT_TEST",
}

FIELDS = [
    "attempt_id", "timestamp", "scenario", "scenario_detail",
    "true_identity", "claimed_identity", "face_score", "behaviour_score",
    "recognized_identity", "runtime_attempt_id", "linkage_verified",
    "label_binding_valid",
    "pad_score", "deepfake_real_score", "v1_fusion_score", "v1_decision",
    "v2_fusion_score", "v2_decision", "v3_fusion_score", "v3_decision",
    "face_reliability", "behaviour_reliability", "pad_reliability",
    "deepfake_reliability", "adaptive_face_weight",
    "adaptive_behaviour_weight", "adaptive_pad_weight",
    "adaptive_deepfake_weight", "calibrated_face_reliability",
    "calibrated_behaviour_reliability", "calibrated_pad_reliability",
    "calibrated_deepfake_reliability", "calibrated_face_weight",
    "calibrated_behaviour_weight", "calibrated_pad_weight",
    "calibrated_deepfake_weight", "consistency_score",
    "detected_conflicts", "attack_severity", "attack_type",
    "security_override", "final_decision", "capture_quality",
    "valid_behaviour_frames", "pad_processed_frames", "pad_skipped_frames",
    "deepfake_processed_frames", "deepfake_skipped_frames", "runtime",
]


def _timestamp():
    return datetime.now(timezone.utc).isoformat()


def _json_value(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def new_attempt_id():
    """Create the identifier shared by one evaluation launch and its runtime log."""
    return uuid.uuid4().hex


def _runtime_record_for_attempt(attempt_id, log_path):
    if not attempt_id:
        raise ValueError("An evaluation attempt ID is required.")
    if not os.path.exists(log_path):
        raise FileNotFoundError(
            f"Runtime fusion log does not exist: {log_path}"
        )
    with open(log_path, encoding="utf-8") as file:
        records = [
            json.loads(line)
            for line in file
            if line.strip()
        ]
    matches = [
        record
        for record in records
        if record.get("attempt_id") == attempt_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one runtime record for attempt {attempt_id}; "
            f"found {len(matches)}."
        )
    return matches[0]


def record_from_runtime_log(
    attempt_id,
    scenario,
    true_identity,
    claimed_identity,
    runtime,
    scenario_detail="",
    log_path="adaptive_fusion/adaptive_fusion_log.jsonl",
):
    scenario = scenario.upper()
    if scenario not in SCENARIOS:
        raise ValueError(f"Unsupported scenario: {scenario}")
    source = _runtime_record_for_attempt(attempt_id, log_path)
    recognized_identity = source.get("recognized_identity")
    identity_labels_present = all(
        isinstance(identity, str) and identity.strip()
        for identity in (true_identity, claimed_identity, recognized_identity)
    )
    label_binding_valid = bool(
        identity_labels_present
        and recognized_identity == true_identity
        and (
            (
                scenario in {"GENUINE", "DEGRADED_CAPTURE"}
                and true_identity == claimed_identity
            )
            or (
                scenario == "IMPOSTOR"
                and true_identity != claimed_identity
            )
        )
    )
    calibrated = source.get("calibrated_reliabilities", {})
    calibrated_weights = source.get("calibrated_weights", {})
    record = {
        "attempt_id": attempt_id,
        "runtime_attempt_id": source.get("attempt_id"),
        "linkage_verified": source.get("attempt_id") == attempt_id,
        "label_binding_valid": label_binding_valid,
        "timestamp": _timestamp(),
        "scenario": scenario,
        "scenario_detail": scenario_detail,
        "true_identity": true_identity,
        "claimed_identity": claimed_identity,
        "recognized_identity": recognized_identity,
        "face_score": source.get("face_score"),
        "behaviour_score": source.get("behaviour_score"),
        "pad_score": source.get("pad_score"),
        "deepfake_real_score": source.get("deepfake_real_score"),
        "v1_fusion_score": source.get("fixed_fusion_score"),
        "v1_decision": source.get("fixed_decision"),
        "v2_fusion_score": source.get("adaptive_fusion_score"),
        "v2_decision": source.get("adaptive_decision"),
        "v3_fusion_score": source.get("calibrated_fusion_score"),
        "v3_decision": source.get("calibrated_decision"),
        "face_reliability": source.get("face_reliability"),
        "behaviour_reliability": source.get("behaviour_reliability"),
        "pad_reliability": source.get("pad_reliability"),
        "deepfake_reliability": source.get("deepfake_reliability"),
        "adaptive_face_weight": source.get("adaptive_weights", {}).get("face"),
        "adaptive_behaviour_weight": source.get("adaptive_weights", {}).get("behaviour"),
        "adaptive_pad_weight": source.get("adaptive_weights", {}).get("pad"),
        "adaptive_deepfake_weight": source.get("adaptive_weights", {}).get("deepfake"),
        "calibrated_face_reliability": calibrated.get("face"),
        "calibrated_behaviour_reliability": calibrated.get("behaviour"),
        "calibrated_pad_reliability": calibrated.get("pad"),
        "calibrated_deepfake_reliability": calibrated.get("deepfake"),
        "calibrated_face_weight": calibrated_weights.get("face"),
        "calibrated_behaviour_weight": calibrated_weights.get("behaviour"),
        "calibrated_pad_weight": calibrated_weights.get("pad"),
        "calibrated_deepfake_weight": calibrated_weights.get("deepfake"),
        "consistency_score": source.get("consistency_score"),
        "detected_conflicts": source.get("detected_conflicts", []),
        "attack_severity": source.get("attack_severity"),
        "attack_type": source.get("attack_type"),
        "security_override": source.get("security_override"),
        "final_decision": source.get("final_decision"),
        "capture_quality": source.get("calibrated_quality_components", {}).get("face", {}).get("capture_quality"),
        "valid_behaviour_frames": source.get("valid_behaviour_frames"),
        "pad_processed_frames": source.get("pad_processed_frames"),
        "pad_skipped_frames": source.get("pad_skipped_frames"),
        "deepfake_processed_frames": source.get("deepfake_processed_frames"),
        "deepfake_skipped_frames": source.get("deepfake_skipped_frames"),
        "runtime": runtime,
    }
    return record


def append_evaluation_record(
    record,
    csv_path=LINKED_EVALUATION_CSV,
    jsonl_path=EVALUATION_JSONL,
):
    missing = [field for field in FIELDS if field not in record]
    if missing:
        raise ValueError(f"Evaluation record missing fields: {missing}")
    json_record = {key: _json_value(record[key]) for key in FIELDS}
    with open(jsonl_path, "a", encoding="utf-8") as file:
        file.write(json.dumps(json_record, sort_keys=True) + "\n")
    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    with open(csv_path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(json_record)
    return json_record


def runtime_seconds_from_output(output):
    marker = "Total program runtime:"
    for line in reversed(output.splitlines()):
        if marker in line:
            return float(line.split(marker, 1)[1].strip().split()[0])
    return None
