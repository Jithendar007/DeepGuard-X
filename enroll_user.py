import os
import time

import cv2
import insightface
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model

from behaviour.realtime_features import RealtimeBehaviourExtractor
from behaviour.preprocessing import normalize_sequence, preprocess_sequence
from user_registry import (
    PROJECT_ROOT,
    UserRegistryError,
    find_user_name,
    load_user_registry,
    next_user_id,
    register_user,
)


FACE_DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
USERS_ROOT = os.path.join(PROJECT_ROOT, "experiments", "users")
NORMALIZATION_FILE = os.path.join(
    PROJECT_ROOT,
    "behaviour",
    "datasets",
    "normalization.npz"
)
BEHAVIOUR_MODEL_FILE = os.path.join(
    PROJECT_ROOT,
    "behaviour",
    "models",
    "behaviour_gru_autoencoder.keras"
)
FACE_CAPTURE_WIDTH = 640
FACE_CAPTURE_HEIGHT = 480
ENROLLMENT_REPETITIONS = 3
ACTION_CAPTURE_DURATION = 1.0
PREPARATION_COUNTDOWN_SECONDS = 2


class EnrollmentCancelled(Exception):
    pass


def sanitize_face_name(name):
    safe_name = "".join(
        character
        for character in name
        if character.isalnum() or character in (" ", "_", "-")
    ).strip()
    if not safe_name:
        raise ValueError("Name contains no usable characters.")
    return safe_name


def capture_face_embedding(face_app, display_name):
    actions = [
        "STRAIGHT / NEUTRAL",
        "NATURAL BLINK",
        "OPEN AND CLOSE MOUTH",
        "SLIGHTLY LEFT",
        "SLIGHTLY RIGHT",
        "RETURN STRAIGHT / NEUTRAL"
    ]
    embeddings = []
    extractor = RealtimeBehaviourExtractor()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FACE_CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FACE_CAPTURE_HEIGHT)

    if not cap.isOpened():
        extractor.close()
        raise RuntimeError("Could not open camera for Face ID enrollment.")

    behaviour_sequences = []
    valid_frames_per_pose = []
    try:
        print(f"\nStarting Face ID enrollment for: {display_name}")
        print("Press Q at any time to cancel.")
        time.sleep(1)

        for repetition in range(ENROLLMENT_REPETITIONS):
            print(f"\nControlled behaviour repetition {repetition + 1}/3")
            repetition_start = len(extractor.get_sequence())
            countdown_started = time.time()
            while time.time() - countdown_started < PREPARATION_COUNTDOWN_SECONDS:
                success, frame = cap.read()
                if not success:
                    continue
                remaining = max(
                    0,
                    int(
                        PREPARATION_COUNTDOWN_SECONDS
                        - (time.time() - countdown_started)
                    ) + 1
                )
                cv2.putText(
                    frame,
                    f"Get ready: {remaining}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )
                cv2.imshow("DeepGuard-X - Enrollment", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    raise EnrollmentCancelled()

            for action_number, instruction in enumerate(actions, start=1):
                print(f"Action {action_number}/6: {instruction}")
                action_started = time.time()
                action_valid_frames = 0
                action_face_captured = False
                while time.time() - action_started < ACTION_CAPTURE_DURATION:
                    success, frame = cap.read()
                    if not success:
                        continue

                    if extractor.process_frame(frame):
                        action_valid_frames += 1

                    if repetition == 0 and not action_face_captured:
                        faces = face_app.get(frame)
                        if len(faces) == 1:
                            face = faces[0]
                            x1, y1, x2, y2 = face.bbox.astype(int)
                            face_width = x2 - x1
                            face_height = y2 - y1
                            cv2.rectangle(
                                frame,
                                (x1, y1),
                                (x2, y2),
                                (0, 255, 0),
                                2
                            )
                            if face_width >= 100 and face_height >= 100 and not action_face_captured:
                                embeddings.append(face.embedding.copy())
                                action_face_captured = True
                                print(f"Action {action_number}/6 face captured")
                                cv2.putText(
                                    frame,
                                    "CAPTURED!",
                                    (20, 120),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    (0, 255, 0),
                                    2
                                )
                            elif not action_face_captured:
                                cv2.putText(
                                    frame,
                                    "Move closer",
                                    (20, 120),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8,
                                    (0, 0, 255),
                                    2
                                )
                        elif len(faces) == 0:
                            cv2.putText(
                                frame,
                                "Face not detected",
                                (20, 120),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 0, 255),
                                2
                            )
                        else:
                            cv2.putText(
                                frame,
                                "Only one person allowed",
                                (20, 120),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 0, 255),
                                2
                            )

                    cv2.putText(
                        frame,
                        f"Action {action_number}/6: {instruction}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2
                    )
                    cv2.putText(
                        frame,
                        f"Time remaining: {max(0.0, ACTION_CAPTURE_DURATION - (time.time() - action_started)):.1f}s",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )
                    cv2.imshow("DeepGuard-X - Enrollment", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        raise EnrollmentCancelled()

                valid_frames_per_pose.append(action_valid_frames)

            repetition_sequence = extractor.get_sequence()[repetition_start:]
            if len(repetition_sequence) < 2:
                raise RuntimeError(
                    f"Behaviour repetition {repetition + 1} produced too few valid frames."
                )
            behaviour_sequences.append(repetition_sequence)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        extractor.close()

    average_embedding = np.mean(np.asarray(embeddings), axis=0)
    norm = np.linalg.norm(average_embedding)
    if norm == 0:
        raise RuntimeError("Face enrollment produced a zero embedding.")
    print(f"Total valid behaviour frames collected: {sum(len(sequence) for sequence in behaviour_sequences)}")
    print(f"Valid behaviour frames per pose: {valid_frames_per_pose}")
    print(f"Raw behaviour sequence shapes: {[np.asarray(sequence).shape for sequence in behaviour_sequences]}")
    return average_embedding / norm, behaviour_sequences


def build_behaviour_profiles(sequences):
    feature_columns = [
        "ear", "mar", "yaw", "pitch", "roll",
        "d_ear", "d_mar", "d_yaw", "d_pitch", "d_roll"
    ]
    normalization = np.load(NORMALIZATION_FILE)
    autoencoder = tf.keras.models.load_model(BEHAVIOUR_MODEL_FILE)
    encoder = Model(
        inputs=autoencoder.input,
        outputs=autoencoder.get_layer("behaviour_embedding").output
    )

    embeddings = []
    stable_means = []
    stable_stds = []
    stable_medians = []
    for sample_number, sequence in enumerate(sequences, start=1):
        rows = []
        for row in sequence:
            values = []
            valid = True
            for feature in feature_columns:
                try:
                    value = float(row[feature])
                except (ValueError, TypeError, KeyError):
                    valid = False
                    break
                if not np.isfinite(value):
                    valid = False
                    break
                values.append(value)
            if valid:
                rows.append(values)

        if len(rows) < 2:
            raise RuntimeError(
                f"Behaviour sample {sample_number} has too few valid feature rows."
            )

        raw_sequence = np.asarray(rows, dtype=np.float32)
        processed_sequence = preprocess_sequence(raw_sequence)
        normalized_sequence = normalize_sequence(
            processed_sequence,
            normalization["mean"],
            normalization["std"]
        )
        embedding = encoder.predict(
            np.expand_dims(normalized_sequence, axis=0),
            verbose=0
        )[0]
        embedding_norm = np.linalg.norm(embedding)
        if embedding_norm > 0:
            embedding = embedding / embedding_norm
        embeddings.append(embedding.astype(np.float32))
        stable_means.append(np.mean(processed_sequence[:, :5], axis=0))
        stable_stds.append(np.std(processed_sequence[:, :5], axis=0))
        stable_medians.append(np.median(processed_sequence[:, :5], axis=0))
        print(f"Behaviour sample {sample_number}: raw={raw_sequence.shape}, processed={processed_sequence.shape}, normalized={normalized_sequence.shape}, embedding={embedding.shape}")

    return (
        np.asarray(embeddings, dtype=np.float32),
        np.median(np.asarray(stable_means), axis=0),
        np.median(np.asarray(stable_stds), axis=0),
        np.median(np.asarray(stable_medians), axis=0)
    )


def save_enrollment(name, user_id, face_embedding, behaviour_data):
    face_folder = os.path.join(FACE_DATA_ROOT, name)
    user_folder = os.path.join(USERS_ROOT, user_id)
    if os.path.exists(face_folder) or os.path.exists(user_folder):
        raise UserRegistryError("Enrollment destination already exists; nothing was overwritten.")

    os.makedirs(face_folder, exist_ok=False)
    os.makedirs(user_folder, exist_ok=False)
    try:
        np.save(os.path.join(face_folder, "embedding.npy"), face_embedding)
        embeddings, stable_mean, stable_std, stable_median = behaviour_data
        profile_embedding = np.mean(embeddings, axis=0)
        profile_norm = np.linalg.norm(profile_embedding)
        if profile_norm > 0:
            profile_embedding = profile_embedding / profile_norm
        np.save(os.path.join(user_folder, "behaviour_embeddings.npy"), embeddings)
        np.save(os.path.join(user_folder, "behaviour_embedding.npy"), profile_embedding)
        np.savez(
            os.path.join(user_folder, "stable_behaviour_profile.npz"),
            mean=stable_mean,
            std=stable_std,
            median=stable_median,
            sample_count=np.array(embeddings.shape[0], dtype=np.int32),
            feature_names=np.array(["ear", "mar", "yaw", "pitch", "roll"])
        )
    except Exception:
        raise


def main():
    entered_name = input("Enter user name: ").strip()
    if not entered_name:
        print("Name cannot be empty.")
        return 1

    try:
        face_name = sanitize_face_name(entered_name)
        registry = load_user_registry()
        existing_name = find_user_name(registry, face_name)
        if existing_name is not None:
            print(
                f"User already exists: {existing_name} -> "
                f"{registry[existing_name]}"
            )
            return 1

        face_folder = os.path.join(FACE_DATA_ROOT, face_name)
        if os.path.exists(face_folder):
            print(f"Face enrollment directory already exists: {face_folder}")
            return 1

        user_id = next_user_id(registry)
        user_folder = os.path.join(USERS_ROOT, user_id)
        if os.path.exists(user_folder):
            print(f"User directory already exists: {user_folder}")
            return 1

        face_app = insightface.app.FaceAnalysis(name="buffalo_l")
        face_app.prepare(ctx_id=-1, det_size=(320, 320))
        face_embedding, behaviour_sequence = capture_face_embedding(
            face_app,
            face_name
        )
        behaviour_data = build_behaviour_profiles(behaviour_sequence)
        save_enrollment(face_name, user_id, face_embedding, behaviour_data)
        register_user(
            face_name,
            user_id,
            directory_already_created=True
        )
    except EnrollmentCancelled:
        print("Enrollment cancelled.")
        return 1
    except (RuntimeError, UserRegistryError, ValueError, OSError) as error:
        print(f"Enrollment failed: {error}")
        return 1

    print("\n======================================")
    print("       ENROLLMENT COMPLETED")
    print("======================================")
    print(f"Face name  : {face_name}")
    print(f"User ID    : {user_id}")
    print(f"Registry   : experiments/users/users.json")
    print("======================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
