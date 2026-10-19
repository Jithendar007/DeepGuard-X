import os
import joblib
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = (
    "experiments/behaviour_sessions/"
    "behaviour_training.csv"
)

MODEL_FOLDER = "behaviour/models"

MODEL_FILE = os.path.join(
    MODEL_FOLDER,
    "behaviour_model.pkl"
)


# ============================================================
# CREATE MODEL FOLDER
# ============================================================

os.makedirs(
    MODEL_FOLDER,
    exist_ok=True
)


# ============================================================
# LOAD DATASET
# ============================================================

if not os.path.exists(INPUT_FILE):

    print("ERROR: Training dataset not found.")
    print(f"Expected: {INPUT_FILE}")
    exit()


df = pd.read_csv(
    INPUT_FILE
)


# ============================================================
# CHECK DATA
# ============================================================

if "session" not in df.columns:

    print(
        "ERROR: 'session' column not found."
    )

    exit()


if len(df) < 5:

    print(
        "ERROR: Too few sessions for training."
    )

    exit()


# ============================================================
# SEPARATE SESSION ID AND FEATURES
# ============================================================

X = df.drop(
    columns=["session"]
)


# Make sure everything is numeric

X = X.apply(
    pd.to_numeric,
    errors="coerce"
)


# Check missing values

if X.isnull().sum().sum() > 0:

    print(
        "ERROR: Missing/non-numeric values "
        "found in training data."
    )

    print(
        X.isnull().sum()[
            X.isnull().sum() > 0
        ]
    )

    exit()


# ============================================================
# DISPLAY TRAINING INFORMATION
# ============================================================

print("\n======================================")
print("       BEHAVIOUR MODEL TRAINING")
print("======================================")

print(
    f"Training sessions : {len(X)}"
)

print(
    f"Features           : {X.shape[1]}"
)

print(
    f"Sessions            : "
    f"{df['session'].tolist()}"
)


# ============================================================
# MODEL PIPELINE
# ============================================================

model = Pipeline([

    (
        "scaler",
        StandardScaler()
    ),

    (
        "isolation_forest",
        IsolationForest(

            n_estimators=200,

            contamination="auto",

            random_state=42

        )
    )

])


# ============================================================
# TRAIN
# ============================================================

print("\nTraining Behaviour Model...")

model.fit(
    X
)


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    model,
    MODEL_FILE
)


# ============================================================
# TRAINING RESULT
# ============================================================

print("\n======================================")
print("       MODEL TRAINING COMPLETED")
print("======================================")

print(
    f"Model saved to:"
)

print(
    MODEL_FILE
)

print(
    f"Training samples : {len(X)}"
)

print(
    f"Features         : {X.shape[1]}"
)

print("======================================")