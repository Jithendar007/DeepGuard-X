"""Honest metrics and summaries for labeled evaluation records."""

import csv
import json
import os
import statistics


REAL_SCENARIOS = {"GENUINE", "IMPOSTOR", "DEGRADED_CAPTURE"}
VERSIONS = {
    "V1": "v1_decision",
    "V2": "v2_decision",
    "V3": "v3_decision",
    "FINAL": "final_decision",
}


def load_records(path="evaluation_results.jsonl"):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as file:
        records = [json.loads(line) for line in file if line.strip()]
    for record in records:
        if "linkage_status" not in record:
            record["linkage_status"] = (
                "LINKED"
                if record.get("linkage_verified") is True
                else "LEGACY_UNLINKED"
            )
    return records


def is_valid_linked_labeled_record(record):
    scenario = record.get("scenario")
    true_identity = record.get("true_identity")
    claimed_identity = record.get("claimed_identity")
    recognized_identity = record.get("recognized_identity")
    attempt_id = record.get("attempt_id")
    return bool(
        record.get("linkage_verified") is True
        and record.get("label_binding_valid") is True
        and isinstance(attempt_id, str)
        and attempt_id.strip()
        and record.get("runtime_attempt_id") == attempt_id
        and all(
            isinstance(identity, str) and identity.strip()
            for identity in (
                true_identity,
                claimed_identity,
                recognized_identity,
            )
        )
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


def _is_genuine(record):
    return record.get("scenario") in {"GENUINE", "DEGRADED_CAPTURE"} and (
        record.get("true_identity") == record.get("claimed_identity")
    )


def _is_impostor(record):
    return record.get("scenario") == "IMPOSTOR" and (
        record.get("true_identity") != record.get("claimed_identity")
    )


def confusion_counts(records, decision_field):
    real_records = [
        record
        for record in records
        if record.get("scenario") in REAL_SCENARIOS
        and is_valid_linked_labeled_record(record)
    ]
    tp = tn = fp = fn = 0
    for record in real_records:
        accepted = record.get(decision_field) == "ACCEPT"
        if _is_genuine(record):
            if accepted:
                tp += 1
            else:
                fn += 1
        elif _is_impostor(record):
            if accepted:
                fp += 1
            else:
                tn += 1
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


def calculate_metrics(counts):
    tp, tn, fp, fn = (counts[key] for key in ("TP", "TN", "FP", "FN"))
    metrics = {}
    metrics["FAR"] = fp / (fp + tn) if fp + tn else None
    metrics["FRR"] = fn / (fn + tp) if fn + tp else None
    metrics["TAR"] = tp / (tp + fn) if tp + fn else None
    metrics["TRR"] = tn / (tn + fp) if tn + fp else None
    total = tp + tn + fp + fn
    metrics["Accuracy"] = (tp + tn) / total if total else None
    return metrics


def _format_metric(value):
    return "N/A - insufficient labeled attempts" if value is None else f"{value:.4f}"


def build_summary(records):
    summary = {}
    for version, field in VERSIONS.items():
        counts = confusion_counts(records, field)
        summary[version] = {
            **counts,
            **calculate_metrics(counts),
            "mean_runtime": (
                statistics.mean(
                    float(record["runtime"])
                    for record in records
                    if record.get("runtime") is not None
                    and record.get("scenario") in REAL_SCENARIOS
                    and is_valid_linked_labeled_record(record)
                )
                if any(
                    record.get("runtime") is not None
                    and record.get("scenario") in REAL_SCENARIOS
                    and is_valid_linked_labeled_record(record)
                    for record in records
                )
                else None
            ),
        }
    return summary


def write_summary(records, output_path="evaluation_summary.csv"):
    summary = build_summary(records)
    fields = ["version", "TP", "TN", "FP", "FN", "FAR", "FRR", "TAR", "TRR", "Accuracy", "mean_runtime"]
    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for version, values in summary.items():
            writer.writerow({"version": version, **values})
    return summary


def write_confusion_text(summary, output_path="confusion_matrices.txt"):
    with open(output_path, "w", encoding="utf-8") as file:
        for version, values in summary.items():
            file.write(f"{version}\n")
            file.write("              Predicted ACCEPT  Predicted REJECT\n")
            file.write(f"Actual genuine       {values['TP']:>5}             {values['FN']:>5}\n")
            file.write(f"Actual impostor      {values['FP']:>5}             {values['TN']:>5}\n\n")


def write_reliability_analysis(records, output_path="reliability_analysis.csv"):
    fields = ["scenario", "count", "face_mean", "behaviour_mean", "pad_mean", "deepfake_mean", "calibrated_face_mean", "calibrated_behaviour_mean", "calibrated_pad_mean", "calibrated_deepfake_mean"]
    linked_records = [
        record for record in records if is_valid_linked_labeled_record(record)
    ]
    scenarios = sorted({record.get("scenario") for record in linked_records})
    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for scenario in scenarios:
            group = [
                record
                for record in linked_records
                if record.get("scenario") == scenario
            ]
            def mean(name):
                values = [float(record[name]) for record in group if record.get(name) is not None]
                return statistics.mean(values) if values else None
            writer.writerow({
                "scenario": scenario,
                "count": len(group),
                "face_mean": mean("face_reliability"),
                "behaviour_mean": mean("behaviour_reliability"),
                "pad_mean": mean("pad_reliability"),
                "deepfake_mean": mean("deepfake_reliability"),
                "calibrated_face_mean": mean("calibrated_face_reliability"),
                "calibrated_behaviour_mean": mean("calibrated_behaviour_reliability"),
                "calibrated_pad_mean": mean("calibrated_pad_reliability"),
                "calibrated_deepfake_mean": mean("calibrated_deepfake_reliability"),
            })


def write_cross_modal_summary(records, output_path="cross_modal_summary.csv"):
    counts = {}
    for record in records:
        if not is_valid_linked_labeled_record(record):
            continue
        severity = record.get("attack_severity") or "UNKNOWN"
        attack_type = record.get("attack_type") or "NONE"
        key = (severity, attack_type, bool(record.get("security_override")))
        counts[key] = counts.get(key, 0) + 1
    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["attack_severity", "attack_type", "security_override", "count"])
        writer.writeheader()
        for (severity, attack_type, override), count in sorted(counts.items()):
            writer.writerow({"attack_severity": severity, "attack_type": attack_type, "security_override": override, "count": count})


def write_robustness_report(records, summary, output_path="ROBUSTNESS_REPORT.md"):
    with open(output_path, "w", encoding="utf-8") as file:
        file.write("# DeepGuard-X Robustness Report\n\n")
        file.write("## 1. Objective\n\nCompare V1 fixed, V2 adaptive, V3 calibrated adaptive, and cross-modal security decisions using recorded attempts.\n\n")
        file.write("## 2. Test Protocol\n\nReal captures use the existing six-action protocol. UNIT_TEST and FAILURE_CASE records are excluded from genuine/impostor metrics.\n\n")
        for heading, scenario in [("Genuine results", "GENUINE"), ("Impostor results", "IMPOSTOR"), ("Degraded capture results", "DEGRADED_CAPTURE"), ("Failure-safe results", "FAILURE_CASE")]:
            file.write(f"## {heading}\n\n")
            group = [
                record
                for record in records
                if record.get("scenario") == scenario
                and is_valid_linked_labeled_record(record)
            ]
            file.write(f"Recorded linked attempts: {len(group)}\n\n")
        legacy_count = sum(
            record.get("linkage_status") == "LEGACY_UNLINKED"
            or not is_valid_linked_labeled_record(record)
            for record in records
        )
        file.write(
            f"Legacy/unlinked or label-invalid records excluded: "
            f"{legacy_count}\n\n"
        )
        file.write("## V1 vs V2 vs V3 Comparison\n\n")
        file.write("| Version | FAR | FRR | TAR | TRR | Accuracy | Mean Runtime |\n|---|---:|---:|---:|---:|---:|---:|\n")
        for version, values in summary.items():
            file.write(f"| {version} | {_format_metric(values['FAR'])} | {_format_metric(values['FRR'])} | {_format_metric(values['TAR'])} | {_format_metric(values['TRR'])} | {_format_metric(values['Accuracy'])} | {_format_metric(values['mean_runtime'])} |\n")
        file.write("\n## Reliability Behaviour\n\nReliability summaries are descriptive measurements. No causality or performance improvement is claimed.\n\n")
        file.write("## Cross-Modal Consistency Behaviour\n\nConflict and security-override counts are written to `cross_modal_summary.csv`.\n\n")
        file.write("## Limitations\n\nInsufficient labeled samples for statistically meaningful FAR/FRR estimation unless enough genuine and impostor attempts are collected.\n\n")
        file.write("## Conclusions\n\nThis report describes measured records only and does not claim that V2 or V3 improves performance.\n")


def generate_reports(path="evaluation_results.jsonl"):
    records = load_records(path)
    summary = write_summary(records)
    write_confusion_text(summary)
    write_reliability_analysis(records)
    write_cross_modal_summary(records)
    write_robustness_report(records, summary)
    return summary
