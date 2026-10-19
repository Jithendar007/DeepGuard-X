"""Compare fixed, current adaptive, and calibrated adaptive fusion logs.

This harness consumes existing JSONL attempt logs. It never stores images or
biometric templates and does not invent labels or classification metrics.
"""

import csv
import json
import os
import sys


LOG_FILE = os.path.join("adaptive_fusion", "adaptive_fusion_log.jsonl")
OUTPUT_JSONL = "fusion_evaluation.jsonl"
OUTPUT_CSV = "fusion_evaluation.csv"


def evaluate_attempt(record):
    required = (
        "face_score",
        "behaviour_score",
        "pad_score",
        "deepfake_real_score",
        "fixed_fusion_score",
        "adaptive_fusion_score",
        "calibrated_fusion_score",
    )
    if not all(name in record for name in required):
        return None
    return {
        "timestamp": record.get("timestamp"),
        "face_score": record["face_score"],
        "behaviour_score": record["behaviour_score"],
        "pad_score": record["pad_score"],
        "deepfake_real_score": record["deepfake_real_score"],
        "v1_score": record["fixed_fusion_score"],
        "v2_score": record["adaptive_fusion_score"],
        "v3_score": record["calibrated_fusion_score"],
        "v1_decision": record.get("fixed_decision"),
        "v2_decision": record.get("adaptive_decision"),
        "v3_decision": record.get("calibrated_decision"),
        "face_reliability": record.get("face_reliability"),
        "behaviour_reliability": record.get("behaviour_reliability"),
        "pad_reliability": record.get("pad_reliability"),
        "deepfake_reliability": record.get("deepfake_reliability"),
        "calibrated_reliabilities": record.get("calibrated_reliabilities"),
        "calibrated_weights": record.get("calibrated_weights"),
        "consistency_score": record.get("consistency_score"),
        "attack_severity": record.get("attack_severity"),
        "attack_type": record.get("attack_type"),
        "final_decision": record.get("final_decision"),
    }


def load_evaluations(path=LOG_FILE):
    if not os.path.exists(path):
        return []
    evaluations = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            evaluation = evaluate_attempt(json.loads(line))
            if evaluation is not None:
                evaluations.append(evaluation)
    return evaluations


def write_outputs(evaluations):
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as file:
        for evaluation in evaluations:
            file.write(json.dumps(evaluation, sort_keys=True) + "\n")

    fields = [
        "timestamp", "face_score", "behaviour_score", "pad_score",
        "deepfake_real_score", "v1_score", "v2_score", "v3_score",
        "v1_decision", "v2_decision", "v3_decision", "face_reliability",
        "behaviour_reliability", "pad_reliability", "deepfake_reliability",
        "calibrated_reliabilities", "calibrated_weights",
        "consistency_score", "attack_severity", "attack_type",
        "final_decision",
    ]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for evaluation in evaluations:
            row = dict(evaluation)
            row["calibrated_reliabilities"] = json.dumps(
                row["calibrated_reliabilities"], sort_keys=True
            )
            row["calibrated_weights"] = json.dumps(
                row["calibrated_weights"], sort_keys=True
            )
            writer.writerow(row)


def main():
    evaluations = load_evaluations(sys.argv[1] if len(sys.argv) > 1 else LOG_FILE)
    write_outputs(evaluations)
    print(f"Evaluated attempts: {len(evaluations)}")
    print(f"JSONL output      : {OUTPUT_JSONL}")
    print(f"CSV output        : {OUTPUT_CSV}")
    print(
        "Insufficient labeled samples for statistically meaningful FAR/FRR estimation."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
