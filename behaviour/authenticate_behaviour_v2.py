import os
import csv
import numpy as np
import tensorflow as tf

from behaviour.preprocessing import normalize_sequence, preprocess_sequence
from tensorflow.keras.models import Model


# ============================================================
# BEHAVIOUR AUTHENTICATION V2
# ============================================================

def authenticate_behaviour(
    user_id,
    test_session=None,
    behaviour_sequence=None
):

    # ========================================================
    # SETTINGS
    # ========================================================

    USER_ID = user_id

    if test_session is not None:
        TEST_SESSION = test_session.upper()
    else:
        TEST_SESSION = None

    ROOT_FOLDER = "experiments/behaviour_sessions"

    MODEL_FILE = (
        "behaviour/models/"
        "behaviour_gru_autoencoder.keras"
    )

    NORMALIZATION_FILE = (
        "behaviour/datasets/"
        "normalization.npz"
    )

    ENROLLED_EMBEDDINGS_FILE = os.path.join(
        "experiments",
        "users",
        USER_ID,
        "behaviour_embeddings.npy"
    )

    LEGACY_EMBEDDING_FILE = os.path.join(
        "experiments",
        "users",
        USER_ID,
        "behaviour_embedding.npy"
    )

    STABLE_PROFILE_FILE = os.path.join(
        "experiments",
        "users",
        USER_ID,
        "stable_behaviour_profile.npz"
    )

    TARGET_FRAMES = 120

    FEATURE_COLUMNS = [
        "ear",
        "mar",
        "yaw",
        "pitch",
        "roll",
        "d_ear",
        "d_mar",
        "d_yaw",
        "d_pitch",
        "d_roll"
    ]

    # ========================================================
    # HEADER
    # ========================================================

    print("\n======================================")
    print("    BEHAVIOUR AUTHENTICATION V2")
    print("======================================")

    print(
        f"User          : {USER_ID}"
    )

    # ========================================================
    # CHECK USER PROFILE
    # ========================================================

    if not os.path.exists(ENROLLED_EMBEDDINGS_FILE) and not os.path.exists(
        LEGACY_EMBEDDING_FILE
    ):

        print(
            "\nERROR: Behaviour embedding not found."
        )

        print(f"Expected: {ENROLLED_EMBEDDINGS_FILE} or {LEGACY_EMBEDDING_FILE}")

        return 0.0

    if not os.path.exists(
        STABLE_PROFILE_FILE
    ):

        print(
            "\nERROR: Stable behaviour profile "
            "not found."
        )

        print(
            f"Expected: {STABLE_PROFILE_FILE}"
        )

        return 0.0

    # ========================================================
    # LOAD BEHAVIOUR SEQUENCE
    # ========================================================

    rows = []

    # --------------------------------------------------------
    # REALTIME SEQUENCE
    # --------------------------------------------------------

    if behaviour_sequence is not None:

        print(
            "\nUsing realtime behaviour sequence..."
        )

        for row in behaviour_sequence:

            values = []

            valid = True

            for feature in FEATURE_COLUMNS:

                try:

                    value = float(
                        row[feature]
                    )

                except (
                    ValueError,
                    TypeError,
                    KeyError
                ):

                    valid = False
                    break

                if not np.isfinite(value):

                    valid = False
                    break

                values.append(value)

            if valid:

                rows.append(values)

    # --------------------------------------------------------
    # CSV SEQUENCE
    # --------------------------------------------------------

    else:

        if TEST_SESSION is None:

            print(
                "\nERROR: Test session required "
                "when realtime sequence is not provided."
            )

            return 0.0

        SEQUENCE_FILE = os.path.join(
            ROOT_FOLDER,
            TEST_SESSION,
            "behaviour_sequence.csv"
        )

        print(
            f"Test session  : {TEST_SESSION}"
        )

        if not os.path.exists(
            SEQUENCE_FILE
        ):

            print(
                "\nERROR: Behaviour sequence "
                "not found."
            )

            print(
                f"Expected: {SEQUENCE_FILE}"
            )

            return 0.0

        with open(
            SEQUENCE_FILE,
            "r"
        ) as file:

            reader = csv.DictReader(
                file
            )

            for row in reader:

                values = []

                valid = True

                for feature in FEATURE_COLUMNS:

                    try:

                        value = float(
                            row[feature]
                        )

                    except (
                        ValueError,
                        TypeError,
                        KeyError
                    ):

                        valid = False
                        break

                    if not np.isfinite(value):

                        valid = False
                        break

                    values.append(value)

                if valid:

                    rows.append(values)

    # ========================================================
    # CHECK SEQUENCE
    # ========================================================

    if len(rows) < 2:

        print(
            "\nERROR: Not enough valid frames."
        )

        return 0.0

    sequence = np.array(
        rows,
        dtype=np.float32
    )

    print(
        f"Frames available : "
        f"{len(sequence)}"
    )

    sequence = preprocess_sequence(sequence)

    # ========================================================
    # LOAD NORMALIZATION
    # ========================================================

    normalization = np.load(
        NORMALIZATION_FILE
    )

    mean = normalization["mean"]

    std = normalization["std"]

    normalized_sequence = normalize_sequence(sequence, mean, std)

    # ========================================================
    # LOAD GRU
    # ========================================================

    print(
        "\nLoading GRU model..."
    )

    autoencoder = tf.keras.models.load_model(
        MODEL_FILE
    )

    encoder = Model(
        inputs=autoencoder.input,
        outputs=autoencoder.get_layer(
            "behaviour_embedding"
        ).output
    )

    # ========================================================
    # CURRENT GRU EMBEDDING
    # ========================================================

    print(
        "Extracting temporal embedding..."
    )

    current_embedding = encoder.predict(
        np.expand_dims(
            normalized_sequence,
            axis=0
        ),
        verbose=0
    )[0]

    # ========================================================
    # NORMALIZE CURRENT EMBEDDING
    # ========================================================

    norm = np.linalg.norm(
        current_embedding
    )

    if norm > 0:

        current_embedding = (
            current_embedding / norm
        )

    # ========================================================
    # LOAD ENROLLED EMBEDDING
    # ========================================================

    embedding_file = (
        ENROLLED_EMBEDDINGS_FILE
        if os.path.exists(ENROLLED_EMBEDDINGS_FILE)
        else LEGACY_EMBEDDING_FILE
    )
    enrolled_embeddings = np.asarray(
        np.load(embedding_file),
        dtype=np.float32
    )
    if enrolled_embeddings.ndim == 1:
        enrolled_embeddings = np.expand_dims(enrolled_embeddings, axis=0)
    if enrolled_embeddings.ndim != 2 or enrolled_embeddings.shape[1] != 32:
        print(f"ERROR: Invalid enrolled embedding shape: {enrolled_embeddings.shape}")
        return 0.0

    embedding_norms = np.linalg.norm(enrolled_embeddings, axis=1, keepdims=True)
    embedding_norms[embedding_norms == 0] = 1.0
    enrolled_embeddings = enrolled_embeddings / embedding_norms

    # ========================================================
    # TEMPORAL SIMILARITY
    # ========================================================

    temporal_similarities = np.dot(enrolled_embeddings, current_embedding)
    temporal_cosine = float(np.max(temporal_similarities))
    print(f"Enrolled embedding samples: {len(temporal_similarities)}")
    for sample_number, similarity in enumerate(temporal_similarities, start=1):
        print(f"Temporal similarity sample {sample_number}: {similarity:.4f}")
    print(f"Selected profile similarity: {temporal_cosine:.4f}")

    temporal_score = (
        temporal_cosine + 1.0
    ) / 2.0

    # ========================================================
    # LOAD STABLE PROFILE
    # ========================================================

    stable_profile = np.load(
        STABLE_PROFILE_FILE
    )

    enrollment_mean = (
        stable_profile["mean"]
    )

    enrollment_std = (
        stable_profile["std"]
    )

    # ========================================================
    # CURRENT STABLE FEATURES
    # ========================================================

    stable_indices = [0, 1]
    current_mean = np.mean(
        sequence[:, stable_indices],
        axis=0
    )

    current_median = np.median(
        sequence[:, stable_indices],
        axis=0
    )

    # ========================================================
    # STABLE SIMILARITY
    # ========================================================

    scales = np.maximum(
        enrollment_std[stable_indices],
        np.array([
            0.05,
            0.05
        ])
    )

    enrollment_median = (
        stable_profile["median"]
        if "median" in stable_profile.files
        else enrollment_mean
    )
    enrollment_mean = enrollment_mean[stable_indices]
    enrollment_median = enrollment_median[stable_indices]
    mean_scores = np.exp(
        -np.abs(current_mean - enrollment_mean) / scales
    )
    median_scores = np.exp(
        -np.abs(current_median - enrollment_median) / scales
    )
    feature_scores = 0.5 * (mean_scores + median_scores)
    stable_score = float(np.clip(np.mean(feature_scores), 0.0, 1.0))
    print("Stable feature diagnostics:")
    feature_names = np.asarray(
        stable_profile["feature_names"]
        if "feature_names" in stable_profile.files
        else FEATURE_COLUMNS[:5]
    )[stable_indices]
    for feature_name, score, difference in zip(
        feature_names,
        feature_scores,
        np.abs(current_median - enrollment_median)
    ):
        print(f"  {feature_name}: score={score:.4f}, median_delta={difference:.4f}")

    # ========================================================
    # FINAL BEHAVIOUR SCORE
    # ========================================================

    final_score = (
        0.60 * stable_score
        +
        0.40 * temporal_score
    )

    # ========================================================
    # DECISION
    # ========================================================

    THRESHOLD = 0.70

    if final_score >= THRESHOLD:

        status = "MATCH"

    else:

        status = "MISMATCH"

    # ========================================================
    # OUTPUT
    # ========================================================

    print("\n======================================")
    print("       BEHAVIOUR RESULT")
    print("======================================")

    print(
        f"User                 : {USER_ID}"
    )

    print(
        f"Frames analyzed      : {TARGET_FRAMES}"
    )

    print("--------------------------------------")

    print(
        f"Stable profile score : "
        f"{stable_score:.4f}"
    )

    print(
        f"Temporal score       : "
        f"{temporal_score:.4f}"
    )

    print(
        f"Selected similarity  : "
        f"{temporal_cosine:.4f}"
    )

    print(
        f"Final Behaviour Score: "
        f"{final_score:.4f}"
    )

    print(
        f"Threshold            : "
        f"{THRESHOLD:.2f}"
    )

    print(
        f"Behaviour Status     : "
        f"{status}"
    )

    print("======================================")

    return float(final_score)


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    user_id = "User_001"

    test_session = "B013"

    score = authenticate_behaviour(
        user_id,
        test_session
    )

    print(
        "\nReturned Behaviour Score:"
    )

    print(
        f"{score:.4f}"
    )