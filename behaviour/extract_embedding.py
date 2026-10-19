import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model


# ============================================================
# SETTINGS
# ============================================================

MODEL_FILE = (
    "behaviour/models/"
    "behaviour_gru_autoencoder.keras"
)

DATASET_FILE = (
    "behaviour/datasets/"
    "X_temporal.npy"
)

OUTPUT_FILE = (
    "behaviour/datasets/"
    "test_embedding.npy"
)


# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(MODEL_FILE):

    print("ERROR: GRU model not found.")
    print(f"Expected: {MODEL_FILE}")
    exit()


if not os.path.exists(DATASET_FILE):

    print("ERROR: Temporal dataset not found.")
    print(f"Expected: {DATASET_FILE}")
    exit()


# ============================================================
# LOAD MODEL
# ============================================================

print("\n======================================")
print("      BEHAVIOUR EMBEDDING TEST")
print("======================================")

print("Loading GRU model...")

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
# LOAD TEMPORAL DATA
# ============================================================

X = np.load(
    DATASET_FILE
).astype(
    np.float32
)


print(
    f"Dataset shape : {X.shape}"
)


# ============================================================
# EXTRACT EMBEDDINGS
# ============================================================

print("\nExtracting embeddings...")

embeddings = encoder.predict(
    X,
    verbose=0
)


# ============================================================
# DISPLAY
# ============================================================

print(
    f"Embedding shape : {embeddings.shape}"
)

print(
    f"Embedding size  : {embeddings.shape[1]}"
)


print("\nFirst session embedding:")

print(
    embeddings[0]
)


# ============================================================
# SAVE
# ============================================================

np.save(
    OUTPUT_FILE,
    embeddings
)


# ============================================================
# FINAL
# ============================================================

print("\n======================================")
print("       EMBEDDING EXTRACTION DONE")
print("======================================")

print(
    f"Saved to: {OUTPUT_FILE}"
)

print("======================================")