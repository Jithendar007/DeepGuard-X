import os
import sys
import csv
import numpy as np


# ============================================================
# SETTINGS
# ============================================================

ROOT_FOLDER = "experiments/behaviour_sessions"

ENROLLMENT_SESSION = "B012"

TEST_SESSION = (
    sys.argv[1].upper()
    if len(sys.argv) > 1
    else "B013"
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


# ============================================================
# LOAD TEMPORAL SEQUENCE
# ============================================================

def load_sequence(session_id):

    file_path = os.path.join(
        ROOT_FOLDER,
        session_id,
        "behaviour_sequence.csv"
    )

    if not os.path.exists(file_path):

        print(
            f"ERROR: Sequence file not found for {session_id}"
        )

        print(
            f"Expected: {file_path}"
        )

        exit()

    rows = []

    with open(
        file_path,
        "r"
    ) as file:

        reader = csv.DictReader(file)

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

    if len(rows) < 2:

        print(
            f"ERROR: Not enough valid frames for {session_id}"
        )

        exit()

    return np.array(
        rows,
        dtype=np.float32
    )


# ============================================================
# RESAMPLE
# ============================================================

def resample_sequence(
    sequence,
    target_length
):

    if len(sequence) == target_length:

        return sequence

    old_indices = np.linspace(
        0,
        len(sequence) - 1,
        len(sequence)
    )

    new_indices = np.linspace(
        0,
        len(sequence) - 1,
        target_length
    )

    result = np.zeros(
        (
            target_length,
            sequence.shape[1]
        ),
        dtype=np.float32
    )

    for i in range(
        sequence.shape[1]
    ):

        result[:, i] = np.interp(
            new_indices,
            old_indices,
            sequence[:, i]
        )

    return result


# ============================================================
# LOAD
# ============================================================

print("\n======================================")
print("    BEHAVIOUR FEATURE COMPARISON")
print("======================================")

print(
    f"Enrollment session : {ENROLLMENT_SESSION}"
)

print(
    f"Test session       : {TEST_SESSION}"
)


enrollment = load_sequence(
    ENROLLMENT_SESSION
)

test = load_sequence(
    TEST_SESSION
)


print(
    f"\nEnrollment frames  : {len(enrollment)}"
)

print(
    f"Test frames        : {len(test)}"
)


# ============================================================
# RESAMPLE
# ============================================================

enrollment = resample_sequence(
    enrollment,
    TARGET_FRAMES
)

test = resample_sequence(
    test,
    TARGET_FRAMES
)


# ============================================================
# FEATURE GROUPS
# ============================================================

groups = {

    "Eye": [
        0,  # EAR
        5   # d_EAR
    ],

    "Mouth": [
        1,  # MAR
        6   # d_MAR
    ],

    "Head": [
        2,  # yaw
        3,  # pitch
        4,  # roll
        7,  # d_yaw
        8,  # d_pitch
        9   # d_roll
    ]

}


# ============================================================
# LOAD NORMALIZATION
# ============================================================

normalization_file = (
    "behaviour/datasets/"
    "normalization.npz"
)

if not os.path.exists(
    normalization_file
):

    print(
        "ERROR: normalization.npz not found."
    )

    exit()


normalization = np.load(
    normalization_file
)

mean = normalization["mean"]

std = normalization["std"]


# ============================================================
# NORMALIZE
# ============================================================

enrollment_normalized = (
    enrollment - mean
) / std

test_normalized = (
    test - mean
) / std


# ============================================================
# FEATURE-LEVEL SIMILARITY
# ============================================================

def calculate_similarity(
    enrollment_data,
    test_data,
    indices
):

    enrollment_vector = (
        enrollment_data[:, indices]
        .flatten()
    )

    test_vector = (
        test_data[:, indices]
        .flatten()
    )

    distance = np.mean(
        np.abs(
            enrollment_vector -
            test_vector
        )
    )

    similarity = np.exp(
        -distance
    )

    return float(
        similarity
    )


# ============================================================
# GROUP SCORES
# ============================================================

eye_score = calculate_similarity(
    enrollment_normalized,
    test_normalized,
    groups["Eye"]
)

mouth_score = calculate_similarity(
    enrollment_normalized,
    test_normalized,
    groups["Mouth"]
)

head_score = calculate_similarity(
    enrollment_normalized,
    test_normalized,
    groups["Head"]
)


# ============================================================
# OVERALL
# ============================================================

overall_score = (

    0.35 * eye_score
    +
    0.25 * mouth_score
    +
    0.40 * head_score

)


# ============================================================
# OUTPUT
# ============================================================

print("\n--------------------------------------")

print(
    f"Eye similarity       : "
    f"{eye_score:.4f}"
)

print(
    f"Mouth similarity     : "
    f"{mouth_score:.4f}"
)

print(
    f"Head similarity      : "
    f"{head_score:.4f}"
)

print(
    f"\nFeature-based score  : "
    f"{overall_score:.4f}"
)


# ============================================================
# FEATURE DIFFERENCES
# ============================================================

print("\n--------------------------------------")
print("Average feature differences:")
print("--------------------------------------")


for index, feature in enumerate(
    FEATURE_COLUMNS
):

    difference = np.mean(
        np.abs(
            enrollment_normalized[:, index]
            -
            test_normalized[:, index]
        )
    )

    print(
        f"{feature:<12}: "
        f"{difference:.4f}"
    )


print("\n======================================")
print("       COMPARISON COMPLETED")
print("======================================")