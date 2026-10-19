import os
import csv
import numpy as np


# ============================================================
# SETTINGS
# ============================================================

USER_ID = "User_001"
ENROLLMENT_SESSION = "B012"

ROOT_FOLDER = "experiments/behaviour_sessions"

OUTPUT_FOLDER = os.path.join(
    "experiments",
    "users",
    USER_ID
)

OUTPUT_FILE = os.path.join(
    OUTPUT_FOLDER,
    "stable_behaviour_profile.npz"
)

SEQUENCE_FILE = os.path.join(
    ROOT_FOLDER,
    ENROLLMENT_SESSION,
    "behaviour_sequence.csv"
)


# ============================================================
# FEATURES
# ============================================================

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
# LOAD SEQUENCE
# ============================================================

if not os.path.exists(SEQUENCE_FILE):

    print("ERROR: Enrollment sequence not found.")
    print(f"Expected: {SEQUENCE_FILE}")
    exit()


rows = []

with open(SEQUENCE_FILE, "r") as file:

    reader = csv.DictReader(file)

    for row in reader:

        values = []

        valid = True

        for feature in FEATURE_COLUMNS:

            try:
                value = float(row[feature])

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

    print("ERROR: Not enough valid frames.")
    exit()


X = np.array(
    rows,
    dtype=np.float32
)


# ============================================================
# CALCULATE STABLE PROFILE
# ============================================================

# We intentionally focus on statistical characteristics,
# rather than storing the exact temporal sequence.

profile_mean = np.mean(
    X,
    axis=0
)

profile_std = np.std(
    X,
    axis=0
)

profile_median = np.median(
    X,
    axis=0
)


# ============================================================
# SELECT STABLE FEATURES
# ============================================================

# EAR and MAR are included as baseline geometry measures.
#
# Head pose is represented mainly by its central tendency,
# rather than exact frame-by-frame movement.

stable_indices = [
    0,  # ear
    1,  # mar
    2,  # yaw
    3,  # pitch
    4   # roll
]


stable_mean = profile_mean[
    stable_indices
]

stable_std = profile_std[
    stable_indices
]

stable_median = profile_median[
    stable_indices
]


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)

np.savez(
    OUTPUT_FILE,

    mean=stable_mean,

    std=stable_std,

    median=stable_median,

    feature_names=np.array([
        "ear",
        "mar",
        "yaw",
        "pitch",
        "roll"
    ])
)


# ============================================================
# OUTPUT
# ============================================================

print("\n======================================")
print("      STABLE BEHAVIOUR PROFILE")
print("======================================")

print(
    f"User       : {USER_ID}"
)

print(
    f"Enrollment : {ENROLLMENT_SESSION}"
)

print(
    f"Frames     : {len(X)}"
)

print("\nStable profile:")

for i, name in enumerate([
    "ear",
    "mar",
    "yaw",
    "pitch",
    "roll"
]):

    print(
        f"{name:<10} "
        f"mean={stable_mean[i]:.4f}  "
        f"std={stable_std[i]:.4f}"
    )


print("\nSaved to:")
print(OUTPUT_FILE)

print("======================================")