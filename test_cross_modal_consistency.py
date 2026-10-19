import math
import unittest

from adaptive_fusion.adaptive_fusion import calculate_adaptive_weights
from calibrated_reliability import calculate_calibrated_reliability
from cross_modal_consistency import (
    AttackSeverity,
    analyze_cross_modal_consistency,
)


class CrossModalConsistencyTests(unittest.TestCase):
    def setUp(self):
        self.valid_match = {
            "face_match": True,
        }

    def analyze(self, face, behaviour, pad, deepfake):
        return analyze_cross_modal_consistency(
            face_score=face,
            behaviour_score=behaviour,
            pad_score=pad,
            deepfake_score=deepfake,
            **self.valid_match,
        )

    def test_all_high_is_normal_and_consistent(self):
        result = self.analyze(0.90, 0.90, 0.90, 0.90)
        self.assertEqual(result.attack_severity, AttackSeverity.NORMAL.value)
        self.assertEqual(result.detected_conflicts, [])
        self.assertGreaterEqual(result.consistency_score, 0.99)
        self.assertFalse(result.security_override)

    def test_identity_liveness_conflict_rejects(self):
        result = self.analyze(0.90, 0.90, 0.40, 0.90)
        self.assertIn("IDENTITY_LIVENESS_CONFLICT", result.detected_conflicts)
        self.assertIn(
            result.attack_severity,
            (AttackSeverity.HIGH_RISK.value, AttackSeverity.CRITICAL.value),
        )
        self.assertEqual(result.final_security_decision, "REJECT")

    def test_identity_deepfake_conflict_rejects(self):
        result = self.analyze(0.90, 0.90, 0.90, 0.40)
        self.assertIn("IDENTITY_DEEPFAKE_CONFLICT", result.detected_conflicts)
        self.assertEqual(result.attack_severity, AttackSeverity.HIGH_RISK.value)
        self.assertTrue(result.security_override)

    def test_behaviour_conflict_warns_without_override(self):
        result = self.analyze(0.90, 0.40, 0.90, 0.90)
        self.assertEqual(
            result.detected_conflicts,
            ["IDENTITY_BEHAVIOUR_CONFLICT"],
        )
        self.assertEqual(result.attack_severity, AttackSeverity.WARNING.value)
        self.assertFalse(result.security_override)

    def test_low_face_rejects(self):
        result = self.analyze(0.40, 0.90, 0.90, 0.90)
        self.assertEqual(result.final_security_decision, "REJECT")
        self.assertTrue(result.security_override)

    def test_pad_veto_rejects(self):
        result = self.analyze(0.90, 0.90, 0.49, 0.90)
        self.assertTrue(result.security_override)
        self.assertEqual(result.final_security_decision, "REJECT")

    def test_invalid_number_fails_closed(self):
        result = self.analyze(math.nan, 0.90, 0.90, 0.90)
        self.assertTrue(result.invalid_input)
        self.assertEqual(result.final_security_decision, "REJECT")

    def test_fixed_fusion_formula(self):
        face, behaviour, pad, deepfake = 0.8, 0.7, 0.9, 0.6
        fixed = (
            0.30 * face
            + 0.25 * behaviour
            + 0.25 * pad
            + 0.20 * deepfake
        )
        self.assertAlmostEqual(fixed, 0.76)

    def test_adaptive_weights_normalize(self):
        weights = calculate_adaptive_weights({
            "face": 0.8,
            "behaviour": 0.7,
            "pad": 0.9,
            "deepfake": 0.6,
        })
        self.assertAlmostEqual(sum(weights.values()), 1.0)

    def test_all_zero_weights_are_unavailable(self):
        self.assertIsNone(calculate_adaptive_weights({
            "face": 0.0,
            "behaviour": 0.0,
            "pad": 0.0,
            "deepfake": 0.0,
        }))

    def test_calibrated_reliability_is_bounded_and_deterministic(self):
        sequence = [{
            "ear": 0.30,
            "mar": 0.05,
            "yaw": 0.0,
            "pitch": 0.0,
            "roll": 0.0,
            "d_ear": 0.01,
            "d_mar": 0.02,
            "d_yaw": 0.1,
            "d_pitch": 0.1,
            "d_roll": 0.1,
        }] * 120
        diagnostics = {
            "scores": [0.80, 0.81, 0.79],
            "processed_frames": 3,
            "skipped_frames": 0,
        }
        kwargs = {
            "face_score": 0.80,
            "behaviour_score": 0.80,
            "pad_score": 0.80,
            "deepfake_score": 0.80,
            "face_scores": [0.80, 0.81, 0.79],
            "behaviour_sequence": sequence,
            "pad_diagnostics": diagnostics,
            "deepfake_diagnostics": diagnostics,
            "captured_frames": [],
        }
        first = calculate_calibrated_reliability(**kwargs)
        second = calculate_calibrated_reliability(**kwargs)
        self.assertEqual(first["reliabilities"], second["reliabilities"])
        self.assertTrue(all(0.0 <= value <= 1.0 for value in first["reliabilities"].values()))

    def test_calibrated_quality_decreases_with_inconsistent_face_scores(self):
        kwargs = {
            "face_score": 0.80,
            "behaviour_score": 0.80,
            "pad_score": 0.80,
            "deepfake_score": 0.80,
            "behaviour_sequence": [],
            "pad_diagnostics": {"scores": [0.8], "processed_frames": 1, "skipped_frames": 0},
            "deepfake_diagnostics": {"scores": [0.8], "processed_frames": 1, "skipped_frames": 0},
            "captured_frames": [],
        }
        stable = calculate_calibrated_reliability(face_scores=[0.80, 0.81, 0.79], **kwargs)
        unstable = calculate_calibrated_reliability(face_scores=[0.10, 0.90, 0.20], **kwargs)
        self.assertLess(
            unstable["reliabilities"]["face"],
            stable["reliabilities"]["face"],
        )


if __name__ == "__main__":
    unittest.main()
