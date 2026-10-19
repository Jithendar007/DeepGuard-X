import os
import sys
import time

import cv2
import numpy as np
import torch


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEEPFAKE_BENCH_DIR = os.path.join(BASE_DIR, "DeepfakeBench")
NETWORKS_DIR = os.path.join(DEEPFAKE_BENCH_DIR, "training")

if NETWORKS_DIR not in sys.path:
    sys.path.insert(0, NETWORKS_DIR)
if os.path.join(NETWORKS_DIR, "networks") not in sys.path:
    sys.path.insert(0, os.path.join(NETWORKS_DIR, "networks"))

from networks.xception import Xception


IMAGE_SIZE = 256
TOP_K = 5
COMPARE_BATCH_INFERENCE = os.environ.get(
    "DEEPGUARD_COMPARE_DEEPFAKE",
    ""
).lower() in {"1", "true", "yes"}
BATCH_COMPARISON_TOLERANCE = 1e-5
LAST_DIAGNOSTICS = {}


def print_deepfake_timing(
    model_initialization_time,
    preprocessing_time,
    batch_inference_time,
    postprocessing_time,
    frame_count,
    total_time
):
    print("\n======================================")
    print("       DEEPFAKE PERFORMANCE")
    print("======================================")
    print(
        f"Model initialization : "
        f"{model_initialization_time:.2f} sec"
    )
    print(f"Preprocessing total  : {preprocessing_time:.2f} sec")
    print(f"Batch inference      : {batch_inference_time:.2f} sec")
    print(f"Post-processing      : {postprocessing_time:.2f} sec")
    print("--------------------------------------")
    print(f"Frame count           : {frame_count:02d}")

    if frame_count:
        average_time = batch_inference_time / frame_count
        print(f"Average/frame         : {average_time:.2f} sec")
    else:
        print("Average/frame         : 0.00 sec")

    print("--------------------------------------")
    print(f"Total Deepfake        : {total_time:.2f} sec")
    print("======================================")


def calculate_real_score(frame_scores):
    scores = np.asarray(frame_scores)
    k = min(TOP_K, len(scores))
    top_k_fake_scores = np.sort(scores)[-k:]
    fake_probability = float(np.mean(top_k_fake_scores))
    return 1.0 - fake_probability


def load_deepfake_model():
    pretrained_path = os.path.join(
        DEEPFAKE_BENCH_DIR,
        "training",
        "pretrained",
        "xception-b5690688.pth"
    )
    detector_weight_path = os.path.join(
        DEEPFAKE_BENCH_DIR,
        "training",
        "weights",
        "xception_best.pth"
    )

    missing_files = [
        path for path in (pretrained_path, detector_weight_path)
        if not os.path.exists(path)
    ]
    if missing_files:
        raise FileNotFoundError(
            "Deepfake weight file(s) not found: "
            + ", ".join(missing_files)
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Deepfake device: {device}")

    model = Xception({
        "mode": "original",
        "num_classes": 2,
        "inc": 3,
        "dropout": False
    })

    state_dict = torch.load(pretrained_path, map_location=device)
    for name, weights in list(state_dict.items()):
        if "pointwise" in name:
            state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)

    state_dict = {
        key: value
        for key, value in state_dict.items()
        if "fc" not in key
    }
    model.load_state_dict(state_dict, strict=False)

    checkpoint = torch.load(detector_weight_path, map_location=device)
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            detector_state = checkpoint["state_dict"]
        elif "model" in checkpoint:
            detector_state = checkpoint["model"]
        elif "backbone" in checkpoint:
            detector_state = checkpoint["backbone"]
        else:
            detector_state = checkpoint
    else:
        detector_state = checkpoint

    clean_state = {}
    for key, value in detector_state.items():
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[len("module."):]
        if new_key.startswith("backbone."):
            new_key = new_key[len("backbone."):]
        clean_state[new_key] = value

    model.load_state_dict(clean_state, strict=False)
    model.to(device)
    model.eval()
    return model, device


def process_frames(frames):
    """Return the DeepfakeBench top-k pooled probability that frames are real."""
    global LAST_DIAGNOSTICS
    if not frames:
        LAST_DIAGNOSTICS = {
            "scores": [],
            "processed_frames": 0,
            "skipped_frames": 0,
        }
        print("Deepfake: no frames supplied.")
        return 0.0

    total_start_time = time.perf_counter()
    model_initialization_start_time = time.perf_counter()
    model, device = load_deepfake_model()
    model_initialization_time = (
        time.perf_counter() - model_initialization_start_time
    )
    frame_scores = []
    skipped_frames = 0
    preprocessing_time = 0.0
    preprocessed_tensors = []

    for image in frames:
        if image is None:
            skipped_frames += 1
            continue

        preprocessing_start_time = time.perf_counter()
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image = (image - 0.5) / 0.5
        image = np.transpose(image, (2, 0, 1))
        tensor = torch.from_numpy(image).unsqueeze(0).to(device)
        preprocessing_time += (
            time.perf_counter() - preprocessing_start_time
        )
        preprocessed_tensors.append(tensor)

    if not preprocessed_tensors:
        LAST_DIAGNOSTICS = {
            "scores": [],
            "processed_frames": 0,
            "skipped_frames": skipped_frames,
        }
        print(
            f"Deepfake: no valid scores obtained; skipped {skipped_frames} frame(s)."
        )
        postprocessing_time = 0.0
        print_deepfake_timing(
            model_initialization_time,
            preprocessing_time,
            0.0,
            postprocessing_time,
            0,
            time.perf_counter() - total_start_time
        )
        return 0.0

    batch = torch.cat(preprocessed_tensors, dim=0)
    inference_start_time = time.perf_counter()
    with torch.no_grad():
        features = model.features(batch)
        logits = model.classifier(features)
        probabilities = torch.softmax(logits, dim=1)
    batch_inference_time = time.perf_counter() - inference_start_time

    frame_scores = probabilities[:, 1].cpu().numpy().tolist()
    LAST_DIAGNOSTICS = {
        "scores": frame_scores,
        "processed_frames": len(frame_scores),
        "skipped_frames": skipped_frames,
    }

    if COMPARE_BATCH_INFERENCE:
        old_frame_scores = []
        comparison_start_time = time.perf_counter()
        with torch.no_grad():
            for tensor in preprocessed_tensors:
                old_features = model.features(tensor)
                old_logits = model.classifier(old_features)
                old_probabilities = torch.softmax(old_logits, dim=1)[0]
                old_frame_scores.append(float(old_probabilities[1]))

        old_real_score = calculate_real_score(old_frame_scores)
        new_real_score = calculate_real_score(frame_scores)
        maximum_prediction_difference = max(
            abs(old_score - new_score)
            for old_score, new_score
            in zip(old_frame_scores, frame_scores)
        )
        score_difference = abs(old_real_score - new_real_score)
        comparison_time = time.perf_counter() - comparison_start_time
        print("\n======================================")
        print("       DEEPFAKE BATCH COMPARISON")
        print("======================================")
        print(f"Maximum prediction difference: {maximum_prediction_difference:.8f}")
        print(f"Old final Real score        : {old_real_score:.8f}")
        print(f"New final Real score        : {new_real_score:.8f}")
        print(f"Final score difference      : {score_difference:.8f}")
        print(f"Comparison time             : {comparison_time:.2f} sec")
        print("======================================")

        if (
            maximum_prediction_difference > BATCH_COMPARISON_TOLERANCE
            or score_difference > BATCH_COMPARISON_TOLERANCE
        ):
            raise RuntimeError(
                "Deepfake batch output differs from per-frame output "
                f"beyond tolerance {BATCH_COMPARISON_TOLERANCE}."
            )

    postprocessing_start_time = time.perf_counter()
    real_probability = calculate_real_score(frame_scores)
    postprocessing_time = time.perf_counter() - postprocessing_start_time

    print(
        f"Deepfake: processed {len(frame_scores)} frame(s), "
        f"skipped {skipped_frames}, Real score {real_probability:.4f}"
    )
    print_deepfake_timing(
        model_initialization_time,
        preprocessing_time,
        batch_inference_time,
        postprocessing_time,
        len(frame_scores),
        time.perf_counter() - total_start_time
    )
    return real_probability
