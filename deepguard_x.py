import math
import os
import time
from concurrent.futures import ThreadPoolExecutor

import cv2

from adaptive_fusion.adaptive_fusion import (
    ADAPTIVE_THRESHOLD,
    BASELINE_WEIGHTS,
    calculate_adaptive_weights,
    calculate_adaptive_fusion,
    log_fusion_attempt,
)
from behaviour.authenticate_behaviour_v2 import authenticate_behaviour
from behaviour.realtime_features import RealtimeBehaviourExtractor
from calibrated_reliability import calculate_calibrated_reliability
from capture_quality import assess_capture_quality
from cross_modal_consistency import analyze_cross_modal_consistency
from deepfake import inference as deepfake_inference
from deepfake.inference import process_frames as process_deepfake_frames
from face_recognition.recognize import FaceIdentifier
from pad import inference as pad_inference
from pad.inference import process_frames as process_pad_frames
from user_registry import UserRegistryError, load_user_registry


VIDEO_DURATION = 7.0
FACE_ID_FPS = 3.0
BEHAVIOUR_FPS = 20.0
FUSION_THRESHOLD = 0.70
MAX_HEAVY_MODEL_FRAMES = 20
PREPARATION_COUNTDOWN_SECONDS = 2.0
ACTION_CAPTURE_DURATION = 1.0
BEHAVIOUR_ACTIONS = [
    "STRAIGHT / NEUTRAL",
    "NATURAL BLINK",
    "OPEN AND CLOSE MOUTH",
    "SLIGHTLY LEFT",
    "SLIGHTLY RIGHT",
    "RETURN STRAIGHT / NEUTRAL"
]

def select_frames(frames, maximum):
    if len(frames) <= maximum:
        return frames

    step = (len(frames) - 1) / (maximum - 1)
    indices = [math.floor(index * step) for index in range(maximum)]
    return [frames[index] for index in indices]


def print_final_result(
    face_score,
    pad_score,
    deepfake_score,
    behaviour_score,
    face_scores=None,
    behaviour_sequence=None,
    captured_frames=None,
    face_match=False,
    behaviour_available=True,
    pad_available=True,
    deepfake_available=True,
    face_quality_detector=None,
    quality_frames=None,
    evaluation_attempt_id=None,
    recognized_identity=None,
):
    # Version 1 fixed fusion remains unchanged for baseline comparison.
    fusion_score = (
        0.30 * face_score
        + 0.25 * behaviour_score
        + 0.25 * pad_score
        + 0.20 * deepfake_score
    )
    fixed_decision = "ACCEPT" if fusion_score >= FUSION_THRESHOLD else "REJECT"
    adaptive_result = calculate_adaptive_fusion(
        face_score=face_score,
        behaviour_score=behaviour_score,
        pad_score=pad_score,
        deepfake_score=deepfake_score,
        face_scores=face_scores,
        behaviour_sequence=behaviour_sequence,
        pad_diagnostics=pad_inference.LAST_DIAGNOSTICS,
        deepfake_diagnostics=deepfake_inference.LAST_DIAGNOSTICS,
        captured_frames=captured_frames,
        face_match=face_match,
        behaviour_available=behaviour_available,
        pad_available=pad_available,
        deepfake_available=deepfake_available,
    )
    cross_modal_result = analyze_cross_modal_consistency(
        face_score=face_score,
        behaviour_score=behaviour_score,
        pad_score=pad_score,
        deepfake_score=deepfake_score,
        face_match=face_match,
    )
    capture_quality_result = assess_capture_quality(
        captured_frames if quality_frames is None else quality_frames,
        face_detector=face_quality_detector,
    )
    capture_quality_result["source_frame_count"] = len(captured_frames or [])
    calibrated_result = calculate_calibrated_reliability(
        face_score=face_score,
        behaviour_score=behaviour_score,
        pad_score=pad_score,
        deepfake_score=deepfake_score,
        face_scores=face_scores,
        behaviour_sequence=behaviour_sequence,
        pad_diagnostics=pad_inference.LAST_DIAGNOSTICS,
        deepfake_diagnostics=deepfake_inference.LAST_DIAGNOSTICS,
        captured_frames=captured_frames,
        capture_quality_result=capture_quality_result,
    )
    calibrated_reliabilities = calibrated_result["reliabilities"]
    calibrated_weights = calculate_adaptive_weights(calibrated_reliabilities)
    calibrated_unavailable = (
        calibrated_weights is None
        or not face_match
        or not behaviour_available
        or not pad_available
        or not deepfake_available
    )
    if calibrated_unavailable:
        calibrated_weights = {
            name: 0.0
            for name in BASELINE_WEIGHTS
        }
        calibrated_score = 0.0
        calibrated_decision = "REJECT"
        calibrated_status = "Calibrated fusion unavailable: no reliable evidence."
    else:
        calibrated_score = (
            calibrated_weights["face"] * face_score
            + calibrated_weights["behaviour"] * behaviour_score
            + calibrated_weights["pad"] * pad_score
            + calibrated_weights["deepfake"] * deepfake_score
        )
        calibrated_decision = (
            "ACCEPT"
            if calibrated_score >= ADAPTIVE_THRESHOLD
            else "REJECT"
        )
        calibrated_status = None
    if pad_score < 0.50 or cross_modal_result.security_override:
        calibrated_decision = "REJECT"
    final_decision = (
        "ACCEPT"
        if calibrated_decision == "ACCEPT"
        else "REJECT"
    )
    adaptive_weights = adaptive_result["adaptive_weights"]
    reliabilities = adaptive_result["reliabilities"]

    print("\n======================================")
    print("       VERSION 1 - FIXED FUSION")
    print("======================================")
    print(f"Face ID score       : {face_score:.4f}")
    print(f"Behaviour score     : {behaviour_score:.4f}")
    print(f"PAD score           : {pad_score:.4f}")
    print(f"Deepfake Real score : {deepfake_score:.4f}")
    print(f"Fusion score        : {fusion_score:.4f}")
    print(f"Fusion threshold    : {FUSION_THRESHOLD:.2f}")
    print("Fixed decision      : " + fixed_decision)

    print("\n======================================")
    print(" VERSION 2 - EXPERIMENTAL ADAPTIVE")
    print("======================================")
    print(f"Face ID reliability       : {reliabilities['face']:.4f}")
    print(f"Behaviour reliability     : {reliabilities['behaviour']:.4f}")
    print(f"PAD reliability           : {reliabilities['pad']:.4f}")
    print(f"Deepfake reliability     : {reliabilities['deepfake']:.4f}")
    print(f"Adaptive Face ID weight   : {adaptive_weights['face']:.4f}")
    print(f"Adaptive Behaviour weight : {adaptive_weights['behaviour']:.4f}")
    print(f"Adaptive PAD weight       : {adaptive_weights['pad']:.4f}")
    print(f"Adaptive Deepfake weight  : {adaptive_weights['deepfake']:.4f}")
    print(f"Adaptive fusion score     : {adaptive_result['adaptive_score']:.4f}")
    print(f"Adaptive threshold        : {ADAPTIVE_THRESHOLD:.2f}")
    if adaptive_result["unavailable_reason"]:
        print(f"Adaptive status            : {adaptive_result['unavailable_reason']}")
    print(
        "PAD security veto         : "
        + ("TRIGGERED" if adaptive_result["pad_veto"] else "NOT TRIGGERED")
    )
    print("Adaptive decision          : " + adaptive_result["adaptive_decision"])

    print("\n======================================")
    print("          CAPTURE QUALITY")
    print("--------------------------------------")
    print(f"Facial Visibility : {capture_quality_result['visibility']:.2f}")
    print(f"Sharpness         : {capture_quality_result['sharpness']:.2f}")
    print(f"Illumination      : {capture_quality_result['illumination']:.2f}")
    print(f"Contrast          : {capture_quality_result['contrast']:.2f}")
    print(f"Exposure          : {capture_quality_result['exposure']:.2f}")
    print(f"Face Size         : {capture_quality_result['face_size']:.2f}")
    print(f"Stability         : {capture_quality_result['stability']:.2f}")
    print(f"Overall Quality   : {capture_quality_result['overall_quality_score']:.2f}")
    if capture_quality_result["invalid_reason"]:
        print(f"Quality status    : {capture_quality_result['invalid_reason']}")

    print("\n======================================")
    print(" VERSION 3 - CALIBRATED ADAPTIVE")
    print("======================================")
    for modality in ("face", "behaviour", "pad", "deepfake"):
        print(f"{modality.title()} quality components:")
        for name, value in calibrated_result["components"][modality].items():
            print(f"  {name:<26}: {value:.4f}")
        print(
            f"{modality.title()} calibrated reliability: "
            f"{calibrated_reliabilities[modality]:.4f}"
        )
    print("Calibrated adaptive weights:")
    for modality, weight in calibrated_weights.items():
        print(f"  {modality.title():<24}: {weight:.4f}")
    print(f"Weight sum                 : {sum(calibrated_weights.values()):.10f}")
    print(f"Calibrated fusion score    : {calibrated_score:.4f}")
    print(f"Calibrated decision        : {calibrated_decision}")
    if calibrated_status:
        print(f"Calibrated status          : {calibrated_status}")

    print("\n======================================")
    print("          FUSION COMPARISON")
    print("======================================")
    print(f"Fixed Fusion Score        : {fusion_score:.4f}")
    print(f"Adaptive Fusion Score     : {adaptive_result['adaptive_score']:.4f}")
    print(f"Calibrated Fusion Score   : {calibrated_score:.4f}")
    print(f"Fixed Decision             : {fixed_decision}")
    print(f"Adaptive Decision         : {adaptive_result['adaptive_decision']}")
    print(f"Calibrated Decision       : {calibrated_decision}")
    print("======================================")

    print("\n======================================")
    print("    CROSS-MODAL CONSISTENCY ANALYSIS")
    print("--------------------------------------")
    print(f"Face state                 : {cross_modal_result.face_state}")
    print(f"Behaviour state            : {cross_modal_result.behaviour_state}")
    print(f"PAD state                  : {cross_modal_result.pad_state}")
    print(f"Deepfake state             : {cross_modal_result.deepfake_state}")
    print(f"Consistency score          : {cross_modal_result.consistency_score:.4f}")
    print(f"Detected conflicts         : {', '.join(cross_modal_result.detected_conflicts) or 'NONE'}")
    print(f"Attack severity            : {cross_modal_result.attack_severity}")
    print(f"Attack type                : {cross_modal_result.attack_type}")
    print(f"Security override          : {cross_modal_result.security_override}")
    print(f"Final security decision    : {cross_modal_result.final_security_decision}")

    print("\n======================================")
    print("       FINAL DEEPGUARD-X DECISION")
    print("======================================")
    print(f"Fusion score               : {calibrated_score:.4f}")
    print(f"Consistency score          : {cross_modal_result.consistency_score:.4f}")
    print(f"Attack severity            : {cross_modal_result.attack_severity}")
    print(f"Final decision             : {final_decision}")
    print("======================================")

    log_fusion_attempt({
        "attempt_id": evaluation_attempt_id,
        "recognized_identity": recognized_identity,
        "face_score": float(face_score),
        "behaviour_score": float(behaviour_score),
        "pad_score": float(pad_score),
        "deepfake_real_score": float(deepfake_score),
        "face_reliability": reliabilities["face"],
        "behaviour_reliability": reliabilities["behaviour"],
        "pad_reliability": reliabilities["pad"],
        "deepfake_reliability": reliabilities["deepfake"],
        "adaptive_weights": adaptive_weights,
        "fixed_fusion_score": float(fusion_score),
        "adaptive_fusion_score": adaptive_result["adaptive_score"],
        "pad_veto": adaptive_result["pad_veto"],
        "fixed_decision": fixed_decision,
        "adaptive_decision": adaptive_result["adaptive_decision"],
        "adaptive_unavailable_reason": adaptive_result["unavailable_reason"],
        "calibrated_reliabilities": calibrated_reliabilities,
        "calibrated_quality_components": calibrated_result["components"],
        "capture_quality": capture_quality_result,
        "calibrated_weights": calibrated_weights,
        "calibrated_fusion_score": float(calibrated_score),
        "calibrated_decision": calibrated_decision,
        "calibrated_unavailable_reason": calibrated_status,
        "consistency_score": cross_modal_result.consistency_score,
        "face_state": cross_modal_result.face_state,
        "behaviour_state": cross_modal_result.behaviour_state,
        "pad_state": cross_modal_result.pad_state,
        "deepfake_state": cross_modal_result.deepfake_state,
        "detected_conflicts": cross_modal_result.detected_conflicts,
        "attack_severity": cross_modal_result.attack_severity,
        "attack_type": cross_modal_result.attack_type,
        "security_override": cross_modal_result.security_override,
        "final_decision": final_decision,
        "valid_behaviour_frames": len(behaviour_sequence),
        "pad_processed_frames": pad_inference.LAST_DIAGNOSTICS.get("processed_frames", 0),
        "pad_skipped_frames": pad_inference.LAST_DIAGNOSTICS.get("skipped_frames", 0),
        "deepfake_processed_frames": deepfake_inference.LAST_DIAGNOSTICS.get("processed_frames", 0),
        "deepfake_skipped_frames": deepfake_inference.LAST_DIAGNOSTICS.get("skipped_frames", 0),
        "runtime": None,
    })


def print_performance_timing(
    face_time,
    behaviour_time,
    pad_time,
    deepfake_time,
    fusion_time,
    post_capture_time
):
    print("\n======================================")
    print("       PERFORMANCE TIMING")
    print("======================================")
    print(f"Face Identification : {face_time:.2f} sec")
    print(f"Behaviour           : {behaviour_time:.2f} sec")
    print(f"PAD                 : {pad_time:.2f} sec")
    print(f"Deepfake            : {deepfake_time:.2f} sec")
    print(f"Fusion              : {fusion_time:.2f} sec")
    print("--------------------------------------")
    print(f"Post-capture total  : {post_capture_time:.2f} sec")
    print("======================================")


def print_parallel_performance(
    pad_time,
    deepfake_time,
    parallel_time,
    sequential_time
):
    improvement = sequential_time - parallel_time
    print("\n======================================")
    print("       PARALLEL PERFORMANCE")
    print("======================================")
    print(f"PAD time                 : {pad_time:.2f} sec")
    print(f"Deepfake time            : {deepfake_time:.2f} sec")
    print(f"Parallel combined time   : {parallel_time:.2f} sec")
    print(f"Previous sequential time : {sequential_time:.2f} sec")
    print(f"Estimated improvement    : {improvement:.2f} sec")
    print("======================================")


def run_timed_stage(processor, frames):
    start_time = time.perf_counter()
    try:
        score = processor(frames)
        error = None
    except Exception as exception:
        score = 0.0
        error = exception
    elapsed_time = time.perf_counter() - start_time
    return score, elapsed_time, error


def main():
    program_start_time = time.perf_counter()
    evaluation_attempt_id = os.environ.get(
        "DEEPGUARD_EVALUATION_ATTEMPT_ID"
    )
    print("\nInitializing DeepGuard-X...")

    try:
        user_registry = load_user_registry()
    except UserRegistryError as error:
        print(f"ERROR: User registry unavailable: {error}")
        user_registry = {}

    try:
        face_identifier = FaceIdentifier()
    except Exception as error:
        print(f"ERROR: Face ID initialization failed: {error}")
        face_identifier = None

    try:
        behaviour_extractor = RealtimeBehaviourExtractor()
    except Exception as error:
        print(f"ERROR: Behaviour initialization failed: {error}")
        behaviour_extractor = None

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        if behaviour_extractor is not None:
            behaviour_extractor.close()
        print_final_result(
            0.0,
            0.0,
            0.0,
            0.0,
            evaluation_attempt_id=evaluation_attempt_id,
        )
        return 1

    captured_frames = []
    captured_frame_elapsed_times = []
    start_time = time.time()
    last_face_process_time = -1.0
    last_behaviour_time = -1.0
    face_interval = 1.0 / FACE_ID_FPS
    behaviour_interval = 1.0 / BEHAVIOUR_FPS
    protocol_duration = (
        PREPARATION_COUNTDOWN_SECONDS
        + len(BEHAVIOUR_ACTIONS) * ACTION_CAPTURE_DURATION
    )
    stage = "PREPARATION"

    print("\n======================================")
    print("        DEEPGUARD-X CAPTURE")
    print("======================================")
    print("Preparation : 2 seconds")
    print("Behaviour   : 6 controlled actions, 1 second each")
    print(f"Duration    : {protocol_duration:.0f} seconds")
    print("Press ESC to stop early.")
    print("--------------------------------------")

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("ERROR: Could not read camera frame.")
                break

            current_time = time.time()
            elapsed = current_time - start_time
            if elapsed >= protocol_duration:
                break

            captured_frames.append(frame.copy())
            captured_frame_elapsed_times.append(elapsed)

            if elapsed < PREPARATION_COUNTDOWN_SECONDS:
                stage = "PREPARATION"
                action_text = (
                    f"Get ready: {max(0, int(PREPARATION_COUNTDOWN_SECONDS - elapsed) + 1)}"
                )
            else:
                action_elapsed = elapsed - PREPARATION_COUNTDOWN_SECONDS
                action_number = int(action_elapsed // ACTION_CAPTURE_DURATION)
                action_text = (
                    f"Action {action_number + 1}/6: "
                    f"{BEHAVIOUR_ACTIONS[action_number]}"
                )
                stage = "BEHAVIOUR ANALYSIS"

            remaining = max(0.0, VIDEO_DURATION - elapsed)
            cv2.putText(
                frame,
                f"Stage: {stage} | {action_text}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )
            cv2.putText(
                frame,
                f"Protocol time remaining: {max(0.0, protocol_duration - elapsed):.1f}s",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            cv2.imshow("DeepGuard-X", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                print("\nCapture stopped by user.")
                break

        for frame, elapsed in zip(
            captured_frames,
            captured_frame_elapsed_times,
        ):
            if elapsed < PREPARATION_COUNTDOWN_SECONDS:
                if (
                    face_identifier is not None
                    and (
                        last_face_process_time < 0
                        or elapsed - last_face_process_time >= face_interval
                    )
                ):
                    last_face_process_time = elapsed
                    try:
                        face_identifier.process_frame(frame)
                    except Exception as error:
                        print(f"ERROR: Face ID frame failed: {error}")
                        face_identifier = None
            elif (
                behaviour_extractor is not None
                and (
                    last_behaviour_time < 0
                    or elapsed - last_behaviour_time >= behaviour_interval
                )
            ):
                last_behaviour_time = elapsed
                try:
                    behaviour_extractor.process_frame(frame)
                except Exception as error:
                    print(f"ERROR: Behaviour frame failed: {error}")
                    behaviour_extractor = None
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if behaviour_extractor is not None:
            behaviour_sequence = behaviour_extractor.get_sequence()
            behaviour_extractor.close()
        else:
            behaviour_sequence = []

    print(f"\nCaptured frames in memory: {len(captured_frames)}")
    print(f"Behaviour feature frames : {len(behaviour_sequence)}")

    post_capture_start_time = time.perf_counter()

    face_start_time = time.perf_counter()
    face_scores = []
    if face_identifier is not None:
        try:
            identified_user, face_score = face_identifier.get_result()
            if identified_user is not None:
                face_scores = list(
                    face_identifier.person_scores.get(identified_user, [])
                )
        except Exception as error:
            print(f"ERROR: Face ID result failed: {error}")
            identified_user, face_score = None, 0.0
    else:
        identified_user, face_score = None, 0.0

    if identified_user is None:
        face_score = 0.0
        print("Face ID: no registered user identified.")
    else:
        print(f"Face ID: {identified_user} ({face_score:.4f})")
    face_time = time.perf_counter() - face_start_time

    behaviour_start_time = time.perf_counter()
    behaviour_score = 0.0
    behaviour_available = False
    behaviour_user = user_registry.get(identified_user)
    if behaviour_user is None:
        if identified_user is None:
            print("Behaviour: skipped because Face ID has no identified user.")
        else:
            print(
                "Behaviour: skipped because Face ID identity is not present "
                "in users.json."
            )
    elif not behaviour_sequence:
        print("Behaviour: no valid feature sequence captured.")
    else:
        try:
            behaviour_score = authenticate_behaviour(
                user_id=behaviour_user,
                behaviour_sequence=behaviour_sequence
            )
            behaviour_available = True
        except Exception as error:
            print(f"ERROR: Behaviour authentication failed: {error}")
    behaviour_time = time.perf_counter() - behaviour_start_time

    detector_frames = select_frames(
        captured_frames,
        MAX_HEAVY_MODEL_FRAMES
    )

    parallel_start_time = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as executor:
        pad_future = executor.submit(
            run_timed_stage,
            process_pad_frames,
            detector_frames
        )
        deepfake_future = executor.submit(
            run_timed_stage,
            process_deepfake_frames,
            detector_frames
        )
        pad_score, pad_time, pad_error = pad_future.result()
        deepfake_score, deepfake_time, deepfake_error = (
            deepfake_future.result()
        )
    parallel_time = time.perf_counter() - parallel_start_time

    if pad_error is not None:
        print(f"ERROR: PAD processing failed: {pad_error}")
    if deepfake_error is not None:
        print(f"ERROR: Deepfake processing failed: {deepfake_error}")

    sequential_time = pad_time + deepfake_time
    print_parallel_performance(
        pad_time,
        deepfake_time,
        parallel_time,
        sequential_time
    )

    fusion_start_time = time.perf_counter()
    print_final_result(
        float(face_score),
        float(pad_score),
        float(deepfake_score),
        float(behaviour_score),
        face_scores=face_scores,
        behaviour_sequence=behaviour_sequence,
        captured_frames=captured_frames,
        face_match=identified_user is not None,
        behaviour_available=behaviour_available,
        pad_available=pad_error is None,
        deepfake_available=deepfake_error is None,
        face_quality_detector=(
            face_identifier.face_app
            if face_identifier is not None
            else None
        ),
        quality_frames=detector_frames,
        evaluation_attempt_id=evaluation_attempt_id,
        recognized_identity=identified_user,
    )
    fusion_time = time.perf_counter() - fusion_start_time
    post_capture_time = time.perf_counter() - post_capture_start_time
    print_performance_timing(
        face_time,
        behaviour_time,
        pad_time,
        deepfake_time,
        fusion_time,
        post_capture_time
    )
    total_runtime = time.perf_counter() - program_start_time
    print(f"Total program runtime: {total_runtime:.2f} sec")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
