import os
import sys

import cv2
import numpy as np
import torch


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAS_DIR = os.path.join(BASE_DIR, "face-anti-spoofing")

if FAS_DIR not in sys.path:
    sys.path.insert(0, FAS_DIR)

from uniface import RetinaFace
from models import MiniFASNetV2
from utils import crop_face, to_tensor, xyxy2xywh


INPUT_SIZE = (80, 80)
SCALE = 2.7
FACE_CONFIDENCE = 0.5
PAD_THRESHOLD = 0.50
LAST_DIAGNOSTICS = {}


def load_pad_model():
    weights_folder = os.path.join(FAS_DIR, "weights")
    weight_files = [
        filename
        for filename in os.listdir(weights_folder)
        if filename.endswith(".pth")
    ]

    if not weight_files:
        raise FileNotFoundError(
            f"No PAD .pth weight found in {weights_folder}"
        )

    weight_path = os.path.join(weights_folder, weight_files[0])
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"PAD weight: {weight_path}")
    print(f"PAD device: {device}")

    model = MiniFASNetV2()
    state_dict = torch.load(
        weight_path,
        map_location=device,
        weights_only=True
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, RetinaFace(), device


def process_frames(frames):
    """Return the mean MiniFASNetV2 probability that frames are real."""
    global LAST_DIAGNOSTICS
    if not frames:
        LAST_DIAGNOSTICS = {
            "scores": [],
            "processed_frames": 0,
            "skipped_frames": 0,
        }
        print("PAD: no frames supplied.")
        return 0.0

    model, detector, device = load_pad_model()
    pad_scores = []
    skipped_frames = 0

    for index, image in enumerate(frames, start=1):
        if image is None:
            skipped_frames += 1
            continue

        faces = detector.detect(image)
        faces = [
            face for face in faces
            if face.confidence >= FACE_CONFIDENCE
        ]

        if len(faces) == 0:
            skipped_frames += 1
            continue

        if len(faces) > 1:
            print(f"PAD frame {index}: multiple faces detected - skipped")
            skipped_frames += 1
            continue

        bbox_xywh = xyxy2xywh(faces[0].bbox).astype(int).tolist()
        face_crop = crop_face(
            image,
            bbox_xywh,
            SCALE,
            INPUT_SIZE[1],
            INPUT_SIZE[0]
        )
        tensor = to_tensor(face_crop).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tensor)
            probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]

        real_score = float(probabilities[1])
        pad_scores.append(real_score)

    if not pad_scores:
        LAST_DIAGNOSTICS = {
            "scores": [],
            "processed_frames": 0,
            "skipped_frames": skipped_frames,
        }
        print(
            f"PAD: no valid scores obtained; skipped {skipped_frames} frame(s)."
        )
        return 0.0

    mean_score = float(np.mean(np.asarray(pad_scores)))
    LAST_DIAGNOSTICS = {
        "scores": pad_scores,
        "processed_frames": len(pad_scores),
        "skipped_frames": skipped_frames,
    }
    print(
        f"PAD: processed {len(pad_scores)} frame(s), "
        f"skipped {skipped_frames}, mean Real score {mean_score:.4f}"
    )
    return mean_score
