import pandas as pd
import numpy as np
import os


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = (
    "experiments/behaviour_sessions/"
    "behaviour_training.csv"
)


# ============================================================
# LOAD DATASET
# ============================================================

if not os.path.exists(INPUT_FILE):

    print("ERROR: Training dataset not found.")
    print(f"Expected: {INPUT_FILE}")
    exit()


df = pd.read_csv(INPUT_FILE)


print("\n======================================")
print("   BEHAVIOUR DATASET INSPECTION")
print("======================================")

print(
    f"Sessions : {len(df)}"
)

print(
    f"Columns  : {len(df.columns)}"
)

print(
    f"Features : {len(df.columns) - 1}"
)


# ============================================================
# SESSION LIST
# ============================================================

print("\n--------------------------------------")
print("Sessions")
print("--------------------------------------")

print(
    df["session"].tolist()
)


# ============================================================
# CHECK DUPLICATE SESSIONS
# ============================================================

duplicates = df["session"].duplicated().sum()

print("\n--------------------------------------")
print("Duplicate sessions")
print("--------------------------------------")

print(
    f"Count : {duplicates}"
)


# ============================================================
# CHECK MISSING VALUES
# ============================================================

missing = df.isnull().sum()

missing_total = missing.sum()

print("\n--------------------------------------")
print("Missing values")
print("--------------------------------------")

if missing_total == 0:

    print("No missing values found. OK")

else:

    print(
        f"Total missing values : {missing_total}"
    )

    print(
        missing[missing > 0]
    )


# ============================================================
# CHECK NUMERIC FEATURES
# ============================================================

feature_columns = [
    column
    for column in df.columns
    if column != "session"
]


non_numeric = []

for column in feature_columns:

    if not pd.api.types.is_numeric_dtype(
        df[column]
    ):

        non_numeric.append(column)


print("\n--------------------------------------")
print("Non-numeric features")
print("--------------------------------------")

if not non_numeric:

    print("All features are numeric. OK")

else:

    print(non_numeric)


# ============================================================
# CHECK INFINITE VALUES
# ============================================================

numeric_df = df[
    feature_columns
]

infinite_count = np.isinf(
    numeric_df.values
).sum()


print("\n--------------------------------------")
print("Infinite values")
print("--------------------------------------")

print(
    f"Count : {infinite_count}"
)


# ============================================================
# FEATURE VARIATION
# ============================================================

print("\n--------------------------------------")
print("Feature variation")
print("--------------------------------------")

constant_features = []

for column in feature_columns:

    unique_count = df[column].nunique()

    if unique_count <= 1:

        constant_features.append(
            column
        )


if constant_features:

    print(
        "Constant features:"
    )

    for feature in constant_features:

        print(
            f"  - {feature}"
        )

else:

    print(
        "No constant features found. OK"
    )


# ============================================================
# FEATURE RANGE
# ============================================================

print("\n======================================")
print("       FEATURE RANGE ANALYSIS")
print("======================================")

summary = df[
    feature_columns
].describe().T


for feature in feature_columns:

    minimum = summary.loc[
        feature,
        "min"
    ]

    maximum = summary.loc[
        feature,
        "max"
    ]

    mean = summary.loc[
        feature,
        "mean"
    ]

    std = summary.loc[
        feature,
        "std"
    ]

    print(
        f"{feature:<30} "
        f"min={minimum:>10.4f}  "
        f"max={maximum:>10.4f}  "
        f"mean={mean:>10.4f}  "
        f"std={std:>10.4f}"
    )


# ============================================================
# CORRELATION CHECK
# ============================================================

print("\n======================================")
print("       HIGH CORRELATION CHECK")
print("======================================")

correlation = df[
    feature_columns
].corr()


high_correlations = []

for i in range(
    len(feature_columns)
):

    for j in range(
        i + 1,
        len(feature_columns)
    ):

        feature_a = feature_columns[i]

        feature_b = feature_columns[j]

        value = correlation.loc[
            feature_a,
            feature_b
        ]

        if abs(value) >= 0.95:

            high_correlations.append(
                (
                    feature_a,
                    feature_b,
                    value
                )
            )


if high_correlations:

    for a, b, value in high_correlations:

        print(
            f"{a} <-> {b} : "
            f"{value:.4f}"
        )

else:

    print(
        "No feature pairs above 0.95 correlation."
    )


# ============================================================
# FINAL
# ============================================================

print("\n======================================")
print("       INSPECTION COMPLETED")
print("======================================")

print(
    "Training dataset is ready for review."
)

print(
    "Do NOT train yet until the results "
    "are checked."
)

print("======================================")