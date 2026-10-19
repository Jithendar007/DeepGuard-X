import os
import numpy as np


# ============================================================
# SETTINGS
# ============================================================

USER_ID = "User_001"

DATASET_FOLDER = "behaviour/datasets"

OUTPUT_FOLDER = os.path.join(
    "experiments",
    "users",
    USER_ID
)

EMBEDDING_FILE = os.path.join(
    OUTPUT_FOLDER,
    "behaviour_embedding.npy"
)

# For now we use B012 as our enrollment/test
# sequence because it already exists.
ENROLLMENT_SESSION = "B012"

SEQUENCE_FILE = os.path.join(
    "experiments",
    "behaviour_sessions",
    ENROLLMENT_SESSION,
    "behaviour_sequence.csv"
)

NORMALIZATION_FILE = os.path.join(
    DATASET_FOLDER,
    "normalization.npz"
)


# ============================================================
# IMPORTANT
# ============================================================

print("\n======================================")
print("       BEHAVIOUR ENROLLMENT")
print("======================================")

print(
    f"User              : {USER_ID}"
)

print(
    f"Enrollment session: {ENROLLMENT_SESSION}"
)


# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(SEQUENCE_FILE):

    print(
        "\nERROR: Behaviour sequence not found."
    )

    print(
        f"Expected: {SEQUENCE_FILE}"
    )

    exit()


if not os.path.exists(NORMALIZATION_FILE):

    print(
        "\nERROR: Normalization file not found."
    )

    print(
        f"Expected: {NORMALIZATION_FILE}"
    )

    exit()


# ============================================================
# LOAD SEQUENCE
# ============================================================

import csv

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

rows = []

with open(
    SEQUENCE_FILE,
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
        "\nERROR: Not enough valid frames."
    )

    exit()


sequence = np.array(
    rows,
    dtype=np.float32
)


print(
    f"Original frames   : {len(sequence)}"
)


# ============================================================
# RESAMPLE TO 120 FRAMES
# ============================================================

TARGET_FRAMES = 120

old_indices = np.linspace(
    0,
    len(sequence) - 1,
    len(sequence)
)

new_indices = np.linspace(
    0,
    len(sequence) - 1,
    TARGET_FRAMES
)

resampled = np.zeros(
    (
        TARGET_FRAMES,
        len(FEATURE_COLUMNS)
    ),
    dtype=np.float32
)

for feature_index in range(
    len(FEATURE_COLUMNS)
):

    resampled[:, feature_index] = np.interp(
        new_indices,
        old_indices,
        sequence[:, feature_index]
    )


sequence = resampled


# ============================================================
# LOAD NORMALIZATION
# ============================================================

normalization = np.load(
    NORMALIZATION_FILE
)

mean = normalization["mean"]

std = normalization["std"]


# ============================================================
# NORMALIZE
# ============================================================

sequence = (
    sequence - mean
) / std


sequence = sequence.astype(
    np.float32
)


# ============================================================
# LOAD TRAINED GRU
# ============================================================

import tensorflow as tf
from tensorflow.keras.models import Model


MODEL_FILE = (
    "behaviour/models/"
    "behaviour_gru_autoencoder.keras"
)


if not os.path.exists(MODEL_FILE):

    print(
        "\nERROR: GRU model not found."
    )

    print(
        f"Expected: {MODEL_FILE}"
    )

    exit()


print(
    "\nLoading GRU model..."
)


autoencoder = tf.keras.models.load_model(
    MODEL_FILE
)


# ============================================================
# CREATE ENCODER
# ============================================================

encoder = Model(

    inputs=autoencoder.input,

    outputs=autoencoder.get_layer(
        "behaviour_embedding"
    ).output

)


# ============================================================
# GENERATE EMBEDDING
# ============================================================

print(
    "Generating behaviour embedding..."
)


embedding = encoder.predict(
    np.expand_dims(
        sequence,
        axis=0
    ),
    verbose=0
)


embedding = embedding[0]


# ============================================================
# NORMALIZE EMBEDDING
# ============================================================

embedding_norm = np.linalg.norm(
    embedding
)


if embedding_norm > 0:

    embedding = (
        embedding /
        embedding_norm
    )


# ============================================================
# SAVE USER PROFILE
# ============================================================

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


np.save(
    EMBEDDING_FILE,
    embedding
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n======================================")
print("       ENROLLMENT COMPLETED")
print("======================================")

print(
    f"User              : {USER_ID}"
)

print(
    f"Frames used       : {TARGET_FRAMES}"
)

print(
    f"Embedding size    : {len(embedding)}"
)

print(
    "\nBehaviour embedding:"
)

print(
    embedding
)

print(
    "\nSaved to:"
)

print(
    EMBEDDING_FILE
)

print("======================================")