import os
import pandas as pd
import numpy as np


# ============================================================
# SETTINGS
# ============================================================

ROOT = "experiments/behaviour_sessions"

TRAINING_FILE = os.path.join(
    ROOT,
    "behaviour_training.csv"
)

TEST_SESSION = "B012"

TEST_FILE = os.path.join(
    ROOT,
    TEST_SESSION,
    "behaviour_session_features.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

training_df = pd.read_csv(
    TRAINING_FILE
)

test_df = pd.read_csv(
    TEST_FILE
)


# Remove session column

feature_columns = [
    c for c in training_df.columns
    if c != "session"
]


X_train = training_df[
    feature_columns
].astype(float)


X_test = test_df[
    feature_columns
].astype(float
)


# ============================================================
# CALCULATE TRAINING STATISTICS
# ============================================================

training_mean = X_train.mean()

training_std = X_train.std(
    ddof=1
)


# Avoid division by zero

training_std = training_std.replace(
    0,
    np.nan
)


# ============================================================
# CALCULATE DIFFERENCE
# ============================================================

test_values = X_test.iloc[0]


difference = (
    test_values
    - training_mean
)


z_scores = (
    difference
    / training_std
)


# Absolute z-score tells us
# how unusual the feature is.

absolute_z = z_scores.abs()


# ============================================================
# CREATE RESULT TABLE
# ============================================================

result = pd.DataFrame({

    "feature": feature_columns,

    "B012_value": [
        test_values[c]
        for c in feature_columns
    ],

    "training_mean": [
        training_mean[c]
        for c in feature_columns
    ],

    "training_std": [
        training_std[c]
        for c in feature_columns
    ],

    "difference": [
        difference[c]
        for c in feature_columns
    ],

    "z_score": [
        z_scores[c]
        for c in feature_columns
    ],

    "absolute_z_score": [
        absolute_z[c]
        for c in feature_columns
    ]

})


# ============================================================
# SORT BY MOST UNUSUAL
# ============================================================

result = result.sort_values(
    "absolute_z_score",
    ascending=False
)


# ============================================================
# DISPLAY
# ============================================================

print("\n======================================")
print("       B012 ANOMALY ANALYSIS")
print("======================================")

print(
    "\nFeatures ranked by how different "
    "B012 is from training sessions:"
)

print("--------------------------------------")

for _, row in result.iterrows():

    print(
        f"{row['feature']:<30} "
        f"B012={row['B012_value']:>10.4f} | "
        f"Mean={row['training_mean']:>10.4f} | "
        f"Z={row['z_score']:>8.2f}"
    )


# ============================================================
# TOP FEATURES
# ============================================================

print("\n======================================")
print("       TOP 5 DIFFERENCES")
print("======================================")

top5 = result.head(5)

for _, row in top5.iterrows():

    print(
        f"{row['feature']:<30} "
        f"Z-score={row['z_score']:.2f}"
    )


# ============================================================
# SAVE REPORT
# ============================================================

output_file = os.path.join(
    ROOT,
    f"{TEST_SESSION}_anomaly_analysis.csv"
)

result.to_csv(
    output_file,
    index=False
)


print("\n======================================")
print("Analysis saved to:")
print(output_file)
print("======================================")