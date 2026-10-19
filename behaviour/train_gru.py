import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    GRU,
    Dense,
    RepeatVector,
    TimeDistributed
)
from tensorflow.keras.callbacks import EarlyStopping


# ============================================================
# SETTINGS
# ============================================================

DATASET_FILE = "behaviour/datasets/X_temporal.npy"

MODEL_FOLDER = "behaviour/models"

MODEL_FILE = os.path.join(
    MODEL_FOLDER,
    "behaviour_gru_autoencoder.keras"
)

EMBEDDING_SIZE = 32

EPOCHS = 100

BATCH_SIZE = 3


# ============================================================
# CREATE MODEL DIRECTORY
# ============================================================

os.makedirs(
    MODEL_FOLDER,
    exist_ok=True
)


# ============================================================
# LOAD DATASET
# ============================================================

if not os.path.exists(DATASET_FILE):

    print(
        "ERROR: Temporal dataset not found."
    )

    print(
        f"Expected: {DATASET_FILE}"
    )

    exit()


X = np.load(
    DATASET_FILE
).astype(
    np.float32
)


print("\n======================================")
print("       BEHAVIOUR GRU TRAINING")
print("======================================")


print(
    f"Dataset shape : {X.shape}"
)


if len(X) < 2:

    print(
        "\nERROR: At least 2 sessions are required."
    )

    exit()


# ============================================================
# INPUT SHAPE
# ============================================================

TIMESTEPS = X.shape[1]

FEATURES = X.shape[2]


print(
    f"Timesteps     : {TIMESTEPS}"
)

print(
    f"Features      : {FEATURES}"
)

print(
    f"Embedding     : {EMBEDDING_SIZE}"
)


# ============================================================
# GRU ENCODER
# ============================================================

inputs = Input(
    shape=(
        TIMESTEPS,
        FEATURES
    ),
    name="behaviour_input"
)


encoder_gru = GRU(
    64,
    return_sequences=False,
    name="encoder_gru"
)(
    inputs
)


embedding = Dense(
    EMBEDDING_SIZE,
    activation=None,
    name="behaviour_embedding"
)(
    encoder_gru
)


# ============================================================
# DECODER
# ============================================================

decoder_input = RepeatVector(
    TIMESTEPS,
    name="repeat_embedding"
)(
    embedding
)


decoder_gru = GRU(
    64,
    return_sequences=True,
    name="decoder_gru"
)(
    decoder_input
)


outputs = TimeDistributed(
    Dense(
        FEATURES
    ),
    name="reconstruction"
)(
    decoder_gru
)


# ============================================================
# AUTOENCODER
# ============================================================

autoencoder = Model(
    inputs,
    outputs,
    name="behaviour_gru_autoencoder"
)


autoencoder.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="mse"

)


# ============================================================
# MODEL SUMMARY
# ============================================================

print("\nModel architecture:\n")

autoencoder.summary()


# ============================================================
# TRAIN
# ============================================================

print("\n======================================")
print("       STARTING TRAINING")
print("======================================")


early_stopping = EarlyStopping(

    monitor="loss",

    patience=10,

    restore_best_weights=True

)


history = autoencoder.fit(

    X,

    X,

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    shuffle=True,

    callbacks=[
        early_stopping
    ],

    verbose=1

)


# ============================================================
# SAVE MODEL
# ============================================================

autoencoder.save(
    MODEL_FILE
)


# ============================================================
# FINAL
# ============================================================

final_loss = history.history[
    "loss"
][-1]


print("\n======================================")
print("       GRU TRAINING COMPLETED")
print("======================================")


print(
    f"Final reconstruction loss : "
    f"{final_loss:.6f}"
)


print(
    f"Model saved to:"
)

print(
    MODEL_FILE
)


print("======================================")