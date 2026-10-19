import json
import os
import tempfile
import unittest

from evaluation_metrics import (
    calculate_metrics,
    confusion_counts,
    is_valid_linked_labeled_record,
    load_records,
)
from evaluation_runner import new_attempt_id, record_from_runtime_log


def runtime_entry(attempt_id, recognized_identity):
    return {
        "attempt_id": attempt_id,
        "recognized_identity": recognized_identity,
        "face_score": 0.8,
        "behaviour_score": 0.7,
        "pad_score": 0.9,
        "deepfake_real_score": 0.8,
        "fixed_fusion_score": 0.8,
        "fixed_decision": "ACCEPT",
        "adaptive_fusion_score": 0.8,
        "adaptive_decision": "ACCEPT",
        "calibrated_fusion_score": 0.8,
        "calibrated_decision": "ACCEPT",
        "adaptive_weights": {},
        "calibrated_reliabilities": {},
        "calibrated_weights": {},
        "detected_conflicts": [],
        "final_decision": "ACCEPT",
    }


class EvaluationLinkageTests(unittest.TestCase):
    def test_attempt_ids_are_unique(self):
        self.assertNotEqual(new_attempt_id(), new_attempt_id())

    def test_runtime_lookup_uses_exact_attempt_id(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "runtime.jsonl")
            with open(path, "w", encoding="utf-8") as file:
                file.write(json.dumps(runtime_entry("first", "Alice")) + "\n")
                file.write(json.dumps(runtime_entry("second", "Bob")) + "\n")

            record = record_from_runtime_log(
                attempt_id="first",
                scenario="GENUINE",
                true_identity="Alice",
                claimed_identity="Alice",
                runtime=3.0,
                log_path=path,
            )

        self.assertEqual(record["attempt_id"], "first")
        self.assertEqual(record["runtime_attempt_id"], "first")
        self.assertEqual(record["recognized_identity"], "Alice")
        self.assertTrue(record["linkage_verified"])
        self.assertTrue(record["label_binding_valid"])
        self.assertEqual(record["face_score"], 0.8)

    def test_missing_or_duplicate_runtime_link_fails_explicitly(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "runtime.jsonl")
            with open(path, "w", encoding="utf-8") as file:
                file.write(json.dumps(runtime_entry("duplicate", "Alice")) + "\n")
                file.write(json.dumps(runtime_entry("duplicate", "Alice")) + "\n")
            with self.assertRaisesRegex(ValueError, "found 2"):
                record_from_runtime_log(
                    attempt_id="duplicate",
                    scenario="GENUINE",
                    true_identity="Alice",
                    claimed_identity="Alice",
                    runtime=3.0,
                    log_path=path,
                )
            with self.assertRaisesRegex(ValueError, "found 0"):
                record_from_runtime_log(
                    attempt_id="not-present",
                    scenario="GENUINE",
                    true_identity="Alice",
                    claimed_identity="Alice",
                    runtime=3.0,
                    log_path=path,
                )

    def test_identity_labels_must_match_recognized_runtime_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "runtime.jsonl")
            with open(path, "w", encoding="utf-8") as file:
                file.write(json.dumps(runtime_entry("run-1", "Alice")) + "\n")
                file.write(json.dumps(runtime_entry("run-2", "Alice")) + "\n")

            genuine = record_from_runtime_log(
                attempt_id="run-1",
                scenario="GENUINE",
                true_identity="Alice",
                claimed_identity="Alice",
                runtime=3.0,
                log_path=path,
            )
            mismatched = record_from_runtime_log(
                attempt_id="run-2",
                scenario="IMPOSTOR",
                true_identity="Bob",
                claimed_identity="Alice",
                runtime=3.0,
                log_path=path,
            )

        self.assertTrue(genuine["label_binding_valid"])
        self.assertTrue(is_valid_linked_labeled_record(genuine))
        self.assertFalse(mismatched["label_binding_valid"])
        self.assertFalse(is_valid_linked_labeled_record(mismatched))

    def test_legacy_records_are_flagged_and_excluded_from_metrics(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "records.jsonl")
            legacy = {
                "scenario": "GENUINE",
                "true_identity": "Alice",
                "claimed_identity": "Alice",
                "final_decision": "ACCEPT",
            }
            linked = {
                "attempt_id": "attempt-1",
                "runtime_attempt_id": "attempt-1",
                "linkage_verified": True,
                "label_binding_valid": True,
                "scenario": "GENUINE",
                "true_identity": "Alice",
                "claimed_identity": "Alice",
                "recognized_identity": "Alice",
                "v3_decision": "ACCEPT",
                "final_decision": "ACCEPT",
            }
            with open(path, "w", encoding="utf-8") as file:
                file.write(json.dumps(legacy) + "\n")
                file.write(json.dumps(linked) + "\n")

            records = load_records(path)

        self.assertEqual(records[0]["linkage_status"], "LEGACY_UNLINKED")
        self.assertEqual(records[1]["linkage_status"], "LINKED")
        counts = confusion_counts(records, "final_decision")
        self.assertEqual(counts, {"TP": 1, "TN": 0, "FP": 0, "FN": 0})

    def test_metric_definitions_are_unchanged_for_linked_trials(self):
        counts = {"TP": 8, "TN": 9, "FP": 1, "FN": 2}
        metrics = calculate_metrics(counts)
        self.assertAlmostEqual(metrics["FAR"], 0.1)
        self.assertAlmostEqual(metrics["FRR"], 0.2)
        self.assertAlmostEqual(metrics["TAR"], 0.8)
        self.assertAlmostEqual(metrics["TRR"], 0.9)
        self.assertAlmostEqual(metrics["Accuracy"], 17 / 20)


if __name__ == "__main__":
    unittest.main()
