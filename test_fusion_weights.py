import unittest

from adaptive_fusion.adaptive_fusion import (
    BASELINE_WEIGHTS,
    V2_BASELINE_WEIGHTS,
    calculate_adaptive_weights,
)


class FusionWeightTests(unittest.TestCase):
    def test_v3_baseline_has_four_equal_weights(self):
        self.assertEqual(
            set(BASELINE_WEIGHTS),
            {"face", "behaviour", "pad", "deepfake"},
        )
        self.assertEqual(
            BASELINE_WEIGHTS,
            {
                "face": 0.25,
                "behaviour": 0.25,
                "pad": 0.25,
                "deepfake": 0.25,
            },
        )
        self.assertEqual(sum(BASELINE_WEIGHTS.values()), 1.0)

    def test_adaptive_weights_are_normalized_and_depend_on_reliability(self):
        equal_reliability = {
            "face": 0.5,
            "behaviour": 0.5,
            "pad": 0.5,
            "deepfake": 0.5,
        }
        varied_reliability = {
            "face": 1.0,
            "behaviour": 0.5,
            "pad": 0.25,
            "deepfake": 0.0,
        }
        equal_weights = calculate_adaptive_weights(equal_reliability)
        varied_weights = calculate_adaptive_weights(varied_reliability)

        self.assertEqual(sum(equal_weights.values()), 1.0)
        self.assertAlmostEqual(sum(varied_weights.values()), 1.0)
        self.assertEqual(
            equal_weights,
            {
                "face": 0.25,
                "behaviour": 0.25,
                "pad": 0.25,
                "deepfake": 0.25,
            },
        )
        self.assertGreater(varied_weights["face"], varied_weights["behaviour"])
        self.assertGreater(varied_weights["behaviour"], varied_weights["pad"])
        self.assertEqual(varied_weights["deepfake"], 0.0)

    def test_zero_total_reliability_fails_safely(self):
        self.assertIsNone(calculate_adaptive_weights({
            "face": 0.0,
            "behaviour": 0.0,
            "pad": 0.0,
            "deepfake": 0.0,
        }))

    def test_v2_baseline_remains_unchanged(self):
        self.assertEqual(
            V2_BASELINE_WEIGHTS,
            {
                "face": 0.30,
                "behaviour": 0.25,
                "pad": 0.25,
                "deepfake": 0.20,
            },
        )
        reliabilities = {
            "face": 1.0,
            "behaviour": 1.0,
            "pad": 1.0,
            "deepfake": 1.0,
        }
        weights = calculate_adaptive_weights(
            reliabilities,
            baseline_weights=V2_BASELINE_WEIGHTS,
        )
        self.assertEqual(weights, V2_BASELINE_WEIGHTS)


if __name__ == "__main__":
    unittest.main()
