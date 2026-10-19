import os
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

BEHAVIOUR_ROOT = "experiments/behaviour_sessions"

OUTPUT_FILE = os.path.join(
    BEHAVIOUR_ROOT,
    "behaviour_training.csv"
)


# ============================================================
# FIND BEHAVIOUR SESSIONS
# ============================================================

if not os.path.exists(BEHAVIOUR_ROOT):

    print("ERROR: Behaviour sessions folder not found.")
    exit()


sessions = []

for name in os.listdir(BEHAVIOUR_ROOT):

    if (
        name.startswith("B")
        and name[1:].isdigit()
    ):

        sessions.append(name)


sessions.sort(
    key=lambda x: int(x[1:])
)


print("\n======================================")
print("   BUILD BEHAVIOUR TRAINING DATASET")
print("======================================")


print(
    f"Sessions found : {len(sessions)}"
)

print(
    f"Sessions       : {', '.join(sessions)}"
)


# ============================================================
# READ SESSION FEATURE VECTORS
# ============================================================

all_features = []

valid_sessions = 0


for session in sessions:

    session_file = os.path.join(
        BEHAVIOUR_ROOT,
        session,
        "behaviour_session_features.csv"
    )


    if not os.path.exists(session_file):

        print(
            f"\nWARNING: {session} feature file missing."
        )

        continue


    try:

        df = pd.read_csv(
            session_file
        )


    except Exception as e:

        print(
            f"\nWARNING: Could not read {session}: {e}"
        )

        continue


    if df.empty:

        print(
            f"\nWARNING: {session} contains no data."
        )

        continue


    # One row should represent one session.
    row = df.iloc[0].copy()

    row["session"] = session

    all_features.append(row)

    valid_sessions += 1


    print(
        f"Loaded {session}"
    )


# ============================================================
# CHECK DATA
# ============================================================

if not all_features:

    print(
        "\nERROR: No valid behaviour sessions found."
    )

    exit()


training_df = pd.DataFrame(
    all_features
)


# Put session name first

columns = [
    "session"
] + [
    c for c in training_df.columns
    if c != "session"
]

training_df = training_df[
    columns
]


# ============================================================
# SAVE DATASET
# ============================================================

training_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# DISPLAY SUMMARY
# ============================================================

print("\n======================================")
print("       DATASET CREATED")
print("======================================")

print(
    f"Valid sessions : {valid_sessions}"
)

print(
    f"Rows           : {training_df.shape[0]}"
)

print(
    f"Features       : {training_df.shape[1] - 1}"
)

print(
    f"Output         : {OUTPUT_FILE}"
)

print("======================================")