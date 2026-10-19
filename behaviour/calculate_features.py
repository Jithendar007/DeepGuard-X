import pandas as pd
import numpy as np
import os
import sys

# ============================================================
# FIND LATEST BEHAVIOUR SESSION
# ============================================================

BEHAVIOUR_ROOT = "experiments/behaviour_sessions"

if not os.path.exists(BEHAVIOUR_ROOT):

    print("ERROR: Behaviour sessions folder not found.")
    exit()


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


# ============================================================
# SELECT SESSION
# ============================================================

if len(sys.argv) > 1:

    SESSION_ID = sys.argv[1].upper()

    if (
        not SESSION_ID.startswith("B")
        or not SESSION_ID[1:].isdigit()
    ):

        print("ERROR: Invalid session ID.")
        print(
            "Example: "
            "python behaviour\\calculate_features.py B002"
        )
        exit()

else:

    # If no session is provided,
    # use the latest session.

    latest_number = max(sessions)

    SESSION_ID = f"B{latest_number:03d}"


SESSION_FOLDER = os.path.join(
    BEHAVIOUR_ROOT,
    SESSION_ID
)


INPUT_CSV = os.path.join(
    SESSION_FOLDER,
    "behaviour_features.csv"
)


OUTPUT_CSV = os.path.join(
    SESSION_FOLDER,
    "behaviour_session_features.csv"
)


# ============================================================
# SETTINGS
# ============================================================

EAR_THRESHOLD = 0.21


# ============================================================
# CHECK INPUT
# ============================================================

if not os.path.exists(INPUT_CSV):

    print(
        "ERROR: Behaviour feature CSV not found."
    )

    print(
        f"Expected: {INPUT_CSV}"
    )

    exit()


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    INPUT_CSV
)


print("\n======================================")
print("      BEHAVIOUR FEATURE ANALYSIS")
print("======================================")

print(
    f"Session         : {SESSION_ID}"
)

print(
    f"Frames available: {len(df)}"
)


if len(df) == 0:

    print(
        "No feature data found."
    )

    exit()


# ============================================================
# EAR FEATURES
# ============================================================

ear = df[
    "average_ear"
].values


mean_ear = np.mean(
    ear
)

std_ear = np.std(
    ear
)

min_ear = np.min(
    ear
)

max_ear = np.max(
    ear
)


# ============================================================
# BLINK DETECTION
# ============================================================

eyes_closed = (
    ear < EAR_THRESHOLD
)


blink_count = 0

in_blink = False

blink_lengths = []

current_blink_length = 0


for closed in eyes_closed:

    if closed:

        if not in_blink:

            in_blink = True

            current_blink_length = 1

        else:

            current_blink_length += 1


    else:

        if in_blink:

            if current_blink_length >= 2:

                blink_count += 1

                blink_lengths.append(
                    current_blink_length
                )

            in_blink = False

            current_blink_length = 0


# Blink ending at final frame

if (
    in_blink
    and current_blink_length >= 2
):

    blink_count += 1

    blink_lengths.append(
        current_blink_length
    )


if blink_lengths:

    average_blink_duration = np.mean(
        blink_lengths
    )

else:

    average_blink_duration = 0.0


# ============================================================
# EAR MOVEMENT
# ============================================================

ear_change = np.diff(
    ear
)

ear_movement = np.mean(
    np.abs(ear_change)
)


# ============================================================
# MAR FEATURES
# ============================================================

mar = df[
    "mar"
].values


mean_mar = np.mean(
    mar
)

std_mar = np.std(
    mar
)

min_mar = np.min(
    mar
)

max_mar = np.max(
    mar
)


mar_change = np.diff(
    mar
)

mar_movement = np.mean(
    np.abs(mar_change)
)


# ============================================================
# HEAD POSE
# ============================================================

yaw = df[
    "yaw"
].values

pitch = df[
    "pitch"
].values

roll = df[
    "roll"
].values


# ============================================================
# YAW
# ============================================================

mean_yaw = np.mean(
    yaw
)

std_yaw = np.std(
    yaw
)

min_yaw = np.min(
    yaw
)

max_yaw = np.max(
    yaw
)


# ============================================================
# PITCH
# ============================================================

mean_pitch = np.mean(
    pitch
)

std_pitch = np.std(
    pitch
)

min_pitch = np.min(
    pitch
)

max_pitch = np.max(
    pitch
)


# ============================================================
# ROLL
# ============================================================

mean_roll = np.mean(
    roll
)

std_roll = np.std(
    roll
)

min_roll = np.min(
    roll
)

max_roll = np.max(
    roll
)


# ============================================================
# HEAD MOVEMENT
# ============================================================

yaw_change = np.diff(
    yaw
)

pitch_change = np.diff(
    pitch
)

roll_change = np.diff(
    roll
)


yaw_movement = np.mean(
    np.abs(yaw_change)
)

pitch_movement = np.mean(
    np.abs(pitch_change)
)

roll_movement = np.mean(
    np.abs(roll_change)
)


head_movement = (
    yaw_movement
    + pitch_movement
    + roll_movement
)


# ============================================================
# CREATE FEATURE VECTOR
# ============================================================

features = {

    # -------------------------
    # Eye behaviour
    # -------------------------

    "mean_ear": mean_ear,

    "std_ear": std_ear,

    "min_ear": min_ear,

    "max_ear": max_ear,

    "blink_count": blink_count,

    "average_blink_duration":
        average_blink_duration,

    "ear_movement":
        ear_movement,


    # -------------------------
    # Mouth behaviour
    # -------------------------

    "mean_mar":
        mean_mar,

    "std_mar":
        std_mar,

    "min_mar":
        min_mar,

    "max_mar":
        max_mar,

    "mar_movement":
        mar_movement,


    # -------------------------
    # Head - Yaw
    # -------------------------

    "mean_yaw":
        mean_yaw,

    "std_yaw":
        std_yaw,

    "min_yaw":
        min_yaw,

    "max_yaw":
        max_yaw,


    # -------------------------
    # Head - Pitch
    # -------------------------

    "mean_pitch":
        mean_pitch,

    "std_pitch":
        std_pitch,

    "min_pitch":
        min_pitch,

    "max_pitch":
        max_pitch,


    # -------------------------
    # Head - Roll
    # -------------------------

    "mean_roll":
        mean_roll,

    "std_roll":
        std_roll,

    "min_roll":
        min_roll,

    "max_roll":
        max_roll,


    # -------------------------
    # Head movement
    # -------------------------

    "yaw_movement":
        yaw_movement,

    "pitch_movement":
        pitch_movement,

    "roll_movement":
        roll_movement,

    "head_movement":
        head_movement
}


# ============================================================
# DISPLAY
# ============================================================

print("\n--------------------------------------")


for name, value in features.items():

    if isinstance(
        value,
        (float, np.floating)
    ):

        print(
            f"{name:<30}: "
            f"{value:.4f}"
        )

    else:

        print(
            f"{name:<30}: "
            f"{value}"
        )


# ============================================================
# SAVE SESSION FEATURE VECTOR
# ============================================================

feature_df = pd.DataFrame(
    [features]
)


feature_df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# FINAL
# ============================================================

print("--------------------------------------")

print(
    "Feature vector saved to:"
)

print(
    OUTPUT_CSV
)

print("======================================")