import json
import os
import tempfile
import unittest

from evaluation_metrics import calculate_metrics, confusion_counts, build_summary
from evaluation_runner import (
    FIELDS,
    append_evaluation_record,
    new_attempt_id,
    record_from_runtime_log,
)


class EvaluationFrameworkTests(unittest.TestCase):
    def make_record(self, scenario, true_identity, claimed_identity, decision="REJECT"):
        record = {field: None for field in FIELDS}
        record.update({
            "attempt_id": "test-attempt",
            "timestamp": "2026-09-12T00:00:00+00:00",
            "scenario": scenario,
            "scenario_detail": "deterministic unit test",
            "true_identity": true_identity,
            "claimed_identity": claimed_identity,
            "recognized_identity": (
                true_identity
                if scenario in {"GENUINE", "DEGRADED_CAPTURE", "IMPOSTOR"}
                else None
            ),
            "runtime_attempt_id": "test-attempt",
            "linkage_verified": True,
            "label_binding_valid": (
                scenario in {"GENUINE", "DEGRADED_CAPTURE"}
                and true_identity == claimed_identity
                or scenario == "IMPOSTOR"
                and true_identity != claimed_identity
            ),
            "v1_decision": decision,
            "v2_decision": decision,
            "v3_decision": decision,
            "final_decision": decision,
            "runtime": 1.0,
        })
        return record

    def test_metric_formulas(self):
        counts = {"TP": 8, "TN": 9, "FP": 1, "FN": 2}
        metrics = calculate_metrics(counts)
        self.assertAlmostEqual(metrics["FAR"], 0.1)
        self.assertAlmostEqual(metrics["FRR"], 0.2)
        self.assertAlmostEqual(metrics["TAR"], 0.8)
        self.assertAlmostEqual(metrics["TRR"], 0.9)
        self.assertAlmostEqual(metrics["Accuracy"], 17 / 20)

    def test_confusion_and_scenario_separation(self):
        records = [
            self.make_record("GENUINE", "A", "A", "ACCEPT"),
            self.make_record("GENUINE", "A", "A", "REJECT"),
            self.make_record("IMPOSTOR", "B", "A", "ACCEPT"),
            self.make_record("IMPOSTOR", "B", "A", "REJECT"),
            self.make_record("FAILURE_CASE", "", "", "REJECT"),
            self.make_record("UNIT_TEST", "", "", "REJECT"),
        ]
        counts = confusion_counts(records, "v1_decision")
        self.assertEqual(counts, {"TP": 1, "TN": 1, "FP": 1, "FN": 1})

    def test_insufficient_metrics_are_none(self):
        self.assertIsNone(calculate_metrics({"TP": 0, "TN": 0, "FP": 0, "FN": 0})["FAR"])

    def test_append_writes_csv_and_jsonl_without_overwrite(self):
        record = self.make_record("UNIT_TEST", "", "", "REJECT")
        with tempfile.TemporaryDirectory() as folder:
            csv_path = os.path.join(folder, "results.csv")
            jsonl_path = os.path.join(folder, "results.jsonl")
            append_evaluation_record(record, csv_path, jsonl_path)
            append_evaluation_record(record, csv_path, jsonl_path)
            self.assertEqual(len(open(jsonl_path, encoding="utf-8").readlines()), 2)
            self.assertEqual(len(open(csv_path, encoding="utf-8").readlines()), 3)

    def test_runtime_log_mapping_and_scenario_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            log_path = os.path.join(folder, "log.jsonl")
            with open(log_path, "w", encoding="utf-8") as file:
                file.write(json.dumps({
                    "attempt_id": "test-attempt",
                    "recognized_identity": "A",
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
                }) + "\n")
            record = record_from_runtime_log(
                attempt_id="test-attempt",
                scenario="GENUINE",
                true_identity="A",
                claimed_identity="A",
                runtime=2.0,
                log_path=log_path,
            )
            self.assertEqual(record["scenario"], "GENUINE")
            self.assertEqual(record["runtime"], 2.0)
            self.assertTrue(record["linkage_verified"])
            self.assertTrue(record["label_binding_valid"])
            with self.assertRaises(ValueError):
                record_from_runtime_log(
                    attempt_id="test-attempt",
                    scenario="UNKNOWN",
                    true_identity="A",
                    claimed_identity="A",
                    runtime=2.0,
                    log_path=log_path,
                )

    def test_attempt_ids_are_unique(self):
        self.assertNotEqual(new_attempt_id(), new_attempt_id())

    def test_v1_v2_v3_are_separate_fields(self):
        record = self.make_record("UNIT_TEST", "", "", "REJECT")
        self.assertIn("v1_fusion_score", record)
        self.assertIn("v2_fusion_score", record)
        self.assertIn("v3_fusion_score", record)


if __name__ == "__main__":
    unittest.main()
