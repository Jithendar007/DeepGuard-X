import unittest

from adaptive_fusion.adaptive_fusion import (
    calculate_adaptive_weights,
)
from calibrated_reliability import (
    DEFAULT_CONFIG,
    calculate_calibrated_reliability,
    estimate_behaviour_quality,
    estimate_deepfake_quality,
    estimate_pad_quality,
)


VALID_CAPTURE_QUALITY = {
    "overall_quality_score": 0.8,
    "valid": True,
}
INVALID_CAPTURE_QUALITY = {
    "overall_quality_score": 0.8,
    "valid": False,
}
VALID_MODEL_DIAGNOSTICS = {
    "scores": [0.78, 0.80, 0.82],
    "processed_frames": 3,
    "skipped_frames": 0,
}
VALID_BEHAVIOUR_SEQUENCE = [{
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
}] * 100


class CalibratedReliabilityTests(unittest.TestCase):
    def test_valid_pad_evidence_receives_meaningful_reliability(self):
        components, reliability = estimate_pad_quality(
            0.8,
            VALID_MODEL_DIAGNOSTICS,
            [],
            capture_quality_result=VALID_CAPTURE_QUALITY,
        )
        self.assertTrue(components["evidence_valid"])
        self.assertGreater(reliability, 0.5)
        self.assertLessEqual(reliability, 1.0)

    def test_missing_pad_predictions_have_zero_reliability(self):
        components, reliability = estimate_pad_quality(
            0.0,
            {"scores": [], "processed_frames": 0, "skipped_frames": 3},
            [],
            capture_quality_result=VALID_CAPTURE_QUALITY,
        )
        self.assertFalse(components["evidence_valid"])
        self.assertEqual(components["score_confidence"], 0.0)
        self.assertEqual(reliability, 0.0)

    def test_missing_deepfake_predictions_have_zero_reliability(self):
        components, reliability = estimate_deepfake_quality(
            0.0,
            {"scores": [], "processed_frames": 0, "skipped_frames": 3},
            [],
            capture_quality_result=VALID_CAPTURE_QUALITY,
        )
        self.assertFalse(components["evidence_valid"])
        self.assertEqual(components["score_confidence"], 0.0)
        self.assertEqual(reliability, 0.0)

    def test_valid_zero_deepfake_score_is_not_treated_as_missing(self):
        zero_diagnostics = {
            "scores": [0.0, 0.0, 0.0],
            "processed_frames": 3,
            "skipped_frames": 0,
        }
        components, reliability = estimate_deepfake_quality(
            0.0,
            zero_diagnostics,
            [],
            capture_quality_result=INVALID_CAPTURE_QUALITY,
        )
        self.assertTrue(components["evidence_valid"])
        self.assertEqual(components["score_confidence"], 1.0)
        self.assertGreater(reliability, 0.0)

    def test_invalid_capture_quality_does_not_contribute_to_reliability(self):
        components, reliability = estimate_pad_quality(
            0.8,
            VALID_MODEL_DIAGNOSTICS,
            [],
            capture_quality_result=INVALID_CAPTURE_QUALITY,
        )
        self.assertFalse(components["capture_quality_valid"])
        self.assertEqual(components["capture_quality"], 0.0)
        expected = (
            0.25 * components["processed_frame_ratio"]
            + 0.25 * components["score_confidence"]
            + 0.30 * components["prediction_consistency"]
        )
        self.assertAlmostEqual(reliability, expected)

    def test_valid_capture_quality_contributes_normally(self):
        invalid_components, invalid_reliability = estimate_pad_quality(
            0.8,
            VALID_MODEL_DIAGNOSTICS,
            [],
            capture_quality_result=INVALID_CAPTURE_QUALITY,
        )
        valid_components, valid_reliability = estimate_pad_quality(
            0.8,
            VALID_MODEL_DIAGNOSTICS,
            [],
            capture_quality_result=VALID_CAPTURE_QUALITY,
        )
        self.assertTrue(valid_components["capture_quality_valid"])
        self.assertEqual(valid_components["capture_quality"], 0.8)
        self.assertAlmostEqual(
            valid_reliability - invalid_reliability,
            DEFAULT_CONFIG.model_weights["capture_quality"] * 0.8,
        )

    def test_behaviour_reliability_includes_capture_quality(self):
        invalid_components, invalid_reliability = estimate_behaviour_quality(
            0.8,
            VALID_BEHAVIOUR_SEQUENCE,
            capture_quality_result=INVALID_CAPTURE_QUALITY,
        )
        valid_components, valid_reliability = estimate_behaviour_quality(
            0.8,
            VALID_BEHAVIOUR_SEQUENCE,
            capture_quality_result=VALID_CAPTURE_QUALITY,
        )
        self.assertTrue(valid_components["capture_quality_valid"])
        self.assertAlmostEqual(
            valid_reliability - invalid_reliability,
            DEFAULT_CONFIG.behaviour_weights["capture_quality"] * 0.8,
        )
        self.assertFalse(invalid_components["capture_quality_valid"])

    def test_missing_modality_evidence_is_not_rescued_by_valid_quality(self):
        result = calculate_calibrated_reliability(
            face_score=0.0,
            behaviour_score=0.8,
            pad_score=0.0,
            deepfake_score=0.0,
            face_scores=[],
            behaviour_sequence=[],
            pad_diagnostics={
                "scores": [],
                "processed_frames": 0,
                "skipped_frames": 3,
            },
            deepfake_diagnostics={
                "scores": [],
                "processed_frames": 0,
                "skipped_frames": 3,
            },
            captured_frames=[],
            capture_quality_result=VALID_CAPTURE_QUALITY,
        )
        self.assertEqual(result["reliabilities"]["face"], 0.0)
        self.assertEqual(result["reliabilities"]["behaviour"], 0.0)
        self.assertEqual(result["reliabilities"]["pad"], 0.0)
        self.assertEqual(result["reliabilities"]["deepfake"], 0.0)

    def test_reliabilities_are_bounded_and_weights_normalize(self):
        diagnostics = {
            "scores": [0.8, 0.81, 0.79],
            "processed_frames": 3,
            "skipped_frames": 1,
        }
        result = calculate_calibrated_reliability(
            face_score=0.8,
            behaviour_score=0.8,
            pad_score=0.8,
            deepfake_score=0.8,
            face_scores=[0.8, 0.81, 0.79],
            behaviour_sequence=VALID_BEHAVIOUR_SEQUENCE,
            pad_diagnostics=diagnostics,
            deepfake_diagnostics=diagnostics,
            captured_frames=[],
            capture_quality_result=VALID_CAPTURE_QUALITY,
        )
        reliabilities = result["reliabilities"]
        self.assertTrue(all(0.0 <= value <= 1.0 for value in reliabilities.values()))
        weights = calculate_adaptive_weights(reliabilities)
        self.assertAlmostEqual(sum(weights.values()), 1.0)

    def test_zero_total_reliability_has_no_adaptive_weights(self):
        self.assertIsNone(calculate_adaptive_weights({
            "face": 0.0,
            "behaviour": 0.0,
            "pad": 0.0,
            "deepfake": 0.0,
        }))


if __name__ == "__main__":
    unittest.main()
