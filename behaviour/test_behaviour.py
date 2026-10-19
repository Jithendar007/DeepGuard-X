import os
import joblib
import pandas as pd
import numpy as np


# ============================================================
# SETTINGS
# ============================================================

MODEL_FILE = (
    "behaviour/models/behaviour_model.pkl"
)

BEHAVIOUR_ROOT = (
    "experiments/behaviour_sessions"
)


# ============================================================
# FIND LATEST SESSION
# ============================================================

sessions = []

for name in os.listdir(BEHAVIOUR_ROOT):

    if (
        name.startswith("B")
        and name[1:].isdigit()
    ):

        sessions.append(
            int(name[1:])
        )


if not sessions:

    print("ERROR: No behaviour sessions found.")
    exit()


latest_number = max(sessions)

SESSION_ID = f"B{latest_number:03d}"

SESSION_FILE = os.path.join(
    BEHAVIOUR_ROOT,
    SESSION_ID,
    "behaviour_session_features.csv"
)


# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(MODEL_FILE):

    print("ERROR: Behaviour model not found.")

    print(
        f"Expected: {MODEL_FILE}"
    )

    exit()


if not os.path.exists(SESSION_FILE):

    print("ERROR: Session feature file not found.")

    print(
        f"Expected: {SESSION_FILE}"
    )

    exit()


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(
    MODEL_FILE
)


# ============================================================
# LOAD SESSION FEATURES
# ============================================================

df = pd.read_csv(
    SESSION_FILE
)


# Remove session column if present

if "session" in df.columns:

    df = df.drop(
        columns=["session"]
    )


# Make sure values are numeric

df = df.apply(
    pd.to_numeric,
    errors="coerce"
)


if df.isnull().sum().sum() > 0:

    print(
        "ERROR: Invalid or missing feature values."
    )

    exit()


# ============================================================
# FEATURE ORDER
# ============================================================

# The model was trained using the columns
# from behaviour_training.csv.
#
# We recreate that exact feature order.

training_file = os.path.join(
    BEHAVIOUR_ROOT,
    "behaviour_training.csv"
)


training_df = pd.read_csv(
    training_file
)


feature_columns = [
    column
    for column in training_df.columns
    if column != "session"
]


# Check that all expected features exist

missing_features = [
    feature
    for feature in feature_columns
    if feature not in df.columns
]


if missing_features:

    print(
        "ERROR: Missing features:"
    )

    print(
        missing_features
    )

    exit()


# Select exact training order

X = df[
    feature_columns
]


# ============================================================
# PREDICTION
# ============================================================

prediction = model.predict(
    X
)[0]


# ============================================================
# ISOLATION FOREST SCORE
# ============================================================

decision_score = model.decision_function(
    X
)[0]


# ============================================================
# RESULT
# ============================================================

print("\n======================================")
print("       BEHAVIOUR AUTHENTICATION")
print("======================================")

print(
    f"Test Session       : {SESSION_ID}"
)

print(
    f"Model Prediction   : {prediction}"
)

print(
    f"Decision Score     : "
    f"{decision_score:.6f}"
)


if prediction == 1:

    status = "NORMAL"

else:

    status = "ANOMALOUS"


print("--------------------------------------")

print(
    f"Behaviour Status   : {status}"
)

print("======================================")