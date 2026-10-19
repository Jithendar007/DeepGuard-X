import os
import csv
import numpy as np

from behaviour.preprocessing import (
    FEATURE_COLUMNS,
    TARGET_FRAMES,
    preprocess_sequence
)


# ============================================================
# SETTINGS
# ============================================================

ROOT_FOLDER = "experiments/behaviour_sessions"
OUTPUT_FOLDER = "behaviour/datasets"

OUTPUT_X = os.path.join(
    OUTPUT_FOLDER,
    "X_temporal.npy"
)

OUTPUT_INFO = os.path.join(
    OUTPUT_FOLDER,
    "session_labels.csv"
)

OUTPUT_NORMALIZATION = os.path.join(
    OUTPUT_FOLDER,
    "normalization.npz"
)

# B012 is reserved for authentication testing
TEST_SESSION = "B012"


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# ============================================================
# FIND SESSIONS
# ============================================================

if not os.path.exists(ROOT_FOLDER):

    print(
        "ERROR: Behaviour sessions folder not found."
    )

    exit()


sessions = []

for name in os.listdir(ROOT_FOLDER):

    if (
        name.startswith("B")
        and name[1:].isdigit()
    ):

        sessions.append(
            int(name[1:])
        )


sessions.sort()


print("\n======================================")
print("     BUILD TEMPORAL DATASET")
print("======================================")

print(
    f"Sessions found : {len(sessions)}"
)

print(
    f"Test session   : {TEST_SESSION}"
)

print(
    "Training sessions exclude the test session."
)


# ============================================================
# LOAD CSV
# ============================================================

def load_sequence(sequence_file):

    rows = []

    with open(
        sequence_file,
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

        return None


    return np.array(
        rows,
        dtype=np.float32
    )


# ============================================================
# LOAD TRAINING SESSIONS
# ============================================================

training_sequences = []

session_information = []


for session_number in sessions:

    session_id = (
        f"B{session_number:03d}"
    )


    # --------------------------------------------------------
    # TEST SESSION
    # --------------------------------------------------------

    if session_id == TEST_SESSION:

        print(
            f"Skipping {session_id} "
            "(reserved for testing)"
        )

        continue


    session_folder = os.path.join(
        ROOT_FOLDER,
        session_id
    )


    sequence_file = os.path.join(
        session_folder,
        "behaviour_sequence.csv"
    )


    if not os.path.exists(
        sequence_file
    ):

        print(
            f"WARNING: {session_id} "
            "sequence file missing."
        )

        continue


    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    try:

        sequence = load_sequence(
            sequence_file
        )

    except Exception as error:

        print(
            f"ERROR reading {session_id}: "
            f"{error}"
        )

        continue


    if sequence is None:

        print(
            f"WARNING: {session_id} "
            "does not contain enough valid frames."
        )

        continue


    original_length = (
        sequence.shape[0]
    )


    # --------------------------------------------------------
    # FIX ANGLE WRAPPING
    # --------------------------------------------------------

    sequence = preprocess_sequence(sequence)


    training_sequences.append(
        sequence
    )


    session_information.append({

        "session": session_id,

        "original_frames":
            original_length,

        "final_frames":
            TARGET_FRAMES

    })


    print(
        f"Loaded {session_id} "
        f"→ {original_length} frames "
        f"→ {TARGET_FRAMES} frames"
    )


# ============================================================
# CHECK
# ============================================================

if not training_sequences:

    print(
        "\nERROR: No valid training sessions."
    )

    exit()


# ============================================================
# CREATE RAW TEMPORAL DATASET
# ============================================================

X_raw = np.stack(
    training_sequences
).astype(
    np.float32
)


# ============================================================
# CALCULATE NORMALIZATION
# ============================================================

print("\nCalculating normalization parameters...")


# Shape:
#
# mean = (10,)
# std  = (10,)

feature_mean = np.mean(
    X_raw,
    axis=(0, 1)
)


feature_std = np.std(
    X_raw,
    axis=(0, 1)
)


# Prevent division by zero

feature_std[
    feature_std < 1e-6
] = 1.0


# ============================================================
# STANDARDIZE
# ============================================================

X = (
    X_raw - feature_mean
) / feature_std


X = X.astype(
    np.float32
)


# ============================================================
# SAVE TEMPORAL DATASET
# ============================================================

np.save(
    OUTPUT_X,
    X
)


# ============================================================
# SAVE NORMALIZATION PARAMETERS
# ============================================================

np.savez(
    OUTPUT_NORMALIZATION,

    mean=feature_mean,

    std=feature_std,

    feature_names=np.array(
        FEATURE_COLUMNS
    )
)


# ============================================================
# SAVE SESSION INFORMATION
# ============================================================

with open(
    OUTPUT_INFO,
    "w",
    newline=""
) as file:

    writer = csv.DictWriter(

        file,

        fieldnames=[
            "session",
            "original_frames",
            "final_frames"
        ]

    )

    writer.writeheader()

    writer.writerows(
        session_information
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n======================================")
print("       DATASET CREATED")
print("======================================")

print(
    f"Training sessions : {X.shape[0]}"
)

print(
    f"Frames / session  : {X.shape[1]}"
)

print(
    f"Features / frame  : {X.shape[2]}"
)


print(
    f"\nDataset shape:"
)

print(
    f"X = {X.shape}"
)


print(
    "\nFeature order:"
)

for index, feature in enumerate(
    FEATURE_COLUMNS
):

    print(
        f"{index:02d} → {feature}"
    )


print(
    "\nNormalization mean:"
)

for feature, value in zip(
    FEATURE_COLUMNS,
    feature_mean
):

    print(
        f"{feature:12s} : {value:.6f}"
    )


print(
    "\nNormalization std:"
)

for feature, value in zip(
    FEATURE_COLUMNS,
    feature_std
):

    print(
        f"{feature:12s} : {value:.6f}"
    )


print(
    "\nSaved:"
)

print(
    f"Temporal data : {OUTPUT_X}"
)

print(
    f"Session info  : {OUTPUT_INFO}"
)

print(
    f"Normalization : {OUTPUT_NORMALIZATION}"
)

print("======================================")