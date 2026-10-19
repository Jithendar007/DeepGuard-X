import unittest

import cv2
import numpy as np

from capture_quality import (
    CAPTURE_QUALITY_WEIGHTS,
    assess_capture_quality,
)


class FakeFaceDetector:
    def __init__(self, boxes_by_frame=None):
        self.boxes_by_frame = boxes_by_frame or {}
        self.calls = 0

    def get(self, frame):
        boxes = self.boxes_by_frame.get(self.calls, [(100, 60, 220, 180, 0.95)])
        self.calls += 1
        return [
            {
                "bbox": box[:4],
                "det_score": box[4] if len(box) > 4 else 0.95,
            }
            for box in boxes
        ]


def make_texture_frame(seed=4, brightness=128):
    rng = np.random.default_rng(seed)
    frame = np.full((240, 320, 3), brightness, dtype=np.uint8)
    low = max(0, brightness - 48)
    high = min(256, brightness + 49)
    texture = rng.integers(low, high, (120, 120, 3), dtype=np.uint8)
    frame[60:180, 100:220] = texture
    return frame


class CaptureQualityTests(unittest.TestCase):
    def assess(self, frames, boxes_by_frame=None):
        return assess_capture_quality(
            frames,
            FakeFaceDetector(boxes_by_frame),
        )

    def assert_scores_bounded(self, result):
        for name in CAPTURE_QUALITY_WEIGHTS:
            self.assertGreaterEqual(result[name], 0.0, name)
            self.assertLessEqual(result[name], 1.0, name)
        self.assertGreaterEqual(result["overall_quality_score"], 0.0)
        self.assertLessEqual(result["overall_quality_score"], 1.0)

    def test_good_quality_frame_has_bounded_quality_scores(self):
        result = self.assess([make_texture_frame(seed=index) for index in range(4)])
        self.assert_scores_bounded(result)
        self.assertTrue(result["valid"])
        self.assertGreater(result["overall_quality_score"], 0.60)
        self.assertAlmostEqual(sum(CAPTURE_QUALITY_WEIGHTS.values()), 1.0)

    def test_dark_frame_scores_poor_illumination_and_exposure(self):
        result = self.assess([make_texture_frame(index, 5) for index in range(4)])
        self.assertLess(result["illumination"], 0.25)
        self.assertLess(result["exposure"], 0.90)
        self.assert_scores_bounded(result)

    def test_very_bright_frame_scores_poor_illumination_and_exposure(self):
        frame = np.full((240, 320, 3), 255, dtype=np.uint8)
        result = self.assess([frame.copy() for _ in range(4)])
        self.assertLess(result["illumination"], 0.05)
        self.assertLess(result["exposure"], 0.10)
        self.assert_scores_bounded(result)

    def test_blurred_frame_has_lower_sharpness_than_textured_frame(self):
        sharp = make_texture_frame()
        blurred = cv2.GaussianBlur(sharp, (31, 31), 0)
        sharp_result = self.assess([sharp.copy() for _ in range(4)])
        blurred_result = self.assess([blurred.copy() for _ in range(4)])
        self.assertLess(blurred_result["sharpness"], sharp_result["sharpness"])

    def test_small_face_reduces_face_size_score(self):
        boxes = {index: [(150, 110, 170, 130, 0.95)] for index in range(4)}
        result = self.assess(
            [make_texture_frame(index) for index in range(4)],
            boxes,
        )
        self.assertEqual(result["face_size"], 0.0)

    def test_no_face_is_explicitly_invalid_and_low_quality(self):
        boxes = {index: [] for index in range(4)}
        result = self.assess(
            [make_texture_frame(index) for index in range(4)],
            boxes,
        )
        self.assertFalse(result["valid"])
        self.assertEqual(result["visibility"], 0.0)
        self.assertEqual(result["overall_quality_score"], 0.0)
        self.assertEqual(result["invalid_reason"], "insufficient_usable_face_frames")

    def test_empty_and_invalid_inputs_are_safe_and_low_quality(self):
        empty = assess_capture_quality([], FakeFaceDetector())
        invalid = assess_capture_quality(
            [None, np.array([], dtype=np.uint8), np.zeros((2, 2, 5), dtype=np.uint8)],
            FakeFaceDetector(),
        )
        for result in (empty, invalid):
            self.assertFalse(result["valid"])
            self.assertEqual(result["overall_quality_score"], 0.0)
            self.assert_scores_bounded(result)

    def test_insufficient_valid_frames_are_explicit(self):
        result = self.assess([make_texture_frame(), None])
        self.assertFalse(result["valid"])
        self.assertEqual(result["overall_quality_score"], 0.0)
        self.assertEqual(result["invalid_reason"], "insufficient_valid_frames")

    def test_multiple_faces_make_frame_unusable(self):
        boxes = {index: [
            (60, 60, 120, 120, 0.95),
            (180, 60, 240, 120, 0.95),
        ] for index in range(4)}
        result = self.assess(
            [make_texture_frame(index) for index in range(4)],
            boxes,
        )
        self.assertEqual(result["usable_face_frames"], 0)
        self.assertEqual(result["overall_quality_score"], 0.0)

    def test_stable_sequence_scores_higher_than_unstable_sequence(self):
        frames = [make_texture_frame(index) for index in range(5)]
        stable = self.assess(frames)
        boxes = {
            index: [
                (
                    10 if index % 2 else 200,
                    10 if index % 2 else 100,
                    130 if index % 2 else 320,
                    130 if index % 2 else 220,
                    0.95,
                )
            ]
            for index in range(5)
        }
        unstable = self.assess(frames, boxes)
        self.assertGreater(stable["stability"], unstable["stability"])

    def test_capture_quality_score_is_passed_to_calibrated_reliability(self):
        from calibrated_reliability import calculate_calibrated_reliability

        capture_quality = {
            **{name: 0.8 for name in CAPTURE_QUALITY_WEIGHTS},
            "overall_quality_score": 0.37,
            "valid": True,
            "frame_count": 4,
            "valid_frame_count": 4,
            "usable_face_frames": 4,
            "invalid_reason": None,
        }
        diagnostics = {
            "scores": [0.8, 0.81, 0.79],
            "processed_frames": 3,
            "skipped_frames": 0,
        }
        result = calculate_calibrated_reliability(
            face_score=0.8,
            behaviour_score=0.8,
            pad_score=0.8,
            deepfake_score=0.8,
            face_scores=[0.8, 0.81, 0.79],
            behaviour_sequence=[],
            pad_diagnostics=diagnostics,
            deepfake_diagnostics=diagnostics,
            captured_frames=[],
            capture_quality_result=capture_quality,
        )
        self.assertEqual(result["capture_quality"]["overall_quality_score"], 0.37)
        self.assertEqual(result["components"]["face"]["capture_quality"], 0.37)
        self.assertEqual(result["components"]["pad"]["capture_quality"], 0.37)
        self.assertEqual(result["components"]["deepfake"]["capture_quality"], 0.37)


if __name__ == "__main__":
    unittest.main()
