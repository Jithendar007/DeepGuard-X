import cv2
import os
import time
import numpy as np


# ==================================================
# SETTINGS
# ==================================================

VIDEO_DURATION = 5          # seconds
TARGET_FPS = 5              # target for 25 selected frames
TOTAL_FRAMES = VIDEO_DURATION * TARGET_FPS


# ==================================================
# BEHAVIOUR SESSION SETUP
# ==================================================

# Behaviour sessions are completely separate
# from V001, V002, etc. used by other experiments.

BEHAVIOUR_ROOT = "experiments/behaviour_sessions"

os.makedirs(
    BEHAVIOUR_ROOT,
    exist_ok=True
)


# ==================================================
# FIND NEXT BEHAVIOUR SESSION
# ==================================================

existing_sessions = []

for name in os.listdir(BEHAVIOUR_ROOT):

    if (
        name.startswith("B")
        and name[1:].isdigit()
    ):

        existing_sessions.append(
            int(name[1:])
        )


if existing_sessions:

    next_number = max(existing_sessions) + 1

else:

    next_number = 1


SESSION_ID = f"B{next_number:03d}"


# ==================================================
# SESSION FOLDERS
# ==================================================

SESSION_FOLDER = os.path.join(
    BEHAVIOUR_ROOT,
    SESSION_ID
)


# ALL continuous frames
# Used by Behaviour Analysis
BEHAVIOUR_FRAMES_FOLDER = os.path.join(
    SESSION_FOLDER,
    "behaviour_frames"
)


# 25 selected frames
# Used by Face ID / PAD / Deepfake
SELECTED_FOLDER = os.path.join(
    SESSION_FOLDER,
    "selected_frames"
)


# Create folders

os.makedirs(
    BEHAVIOUR_FRAMES_FOLDER,
    exist_ok=True
)

os.makedirs(
    SELECTED_FOLDER,
    exist_ok=True
)


# ==================================================
# SESSION INFORMATION
# ==================================================

print("\n======================================")
print("      DEEPGUARD-X BEHAVIOUR CAPTURE")
print("======================================")

print(
    f"Behaviour Session : {SESSION_ID}"
)

print(
    f"Duration          : {VIDEO_DURATION} seconds"
)

print(
    f"Target FPS        : {TARGET_FPS}"
)

print(
    f"Selected frames   : {TOTAL_FRAMES}"
)

print(
    f"Session folder    : {SESSION_FOLDER}"
)

print("======================================")


# ==================================================
# OPEN WEBCAM
# ==================================================

cap = cv2.VideoCapture(0)

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    640
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    480
)


if not cap.isOpened():

    print(
        "ERROR: Could not open webcam."
    )

    exit()


# ==================================================
# PHASE 1 — CONTINUOUS CAPTURE
# ==================================================

print("\nLook at the camera...")
print("Capture starting...\n")


# Store ALL captured frames in memory

captured_frames = []

start_time = time.time()


while True:

    success, frame = cap.read()


    if not success:

        print(
            "ERROR: Could not read camera."
        )

        break


    # Calculate elapsed time

    elapsed = time.time() - start_time


    # --------------------------------------------------
    # STORE EVERY CAPTURED FRAME
    #
    # These N frames are used for Behaviour Analysis.
    # --------------------------------------------------

    captured_frames.append(
        frame.copy()
    )


    # --------------------------------------------------
    # DISPLAY COUNTDOWN
    # --------------------------------------------------

    remaining = max(
        0,
        VIDEO_DURATION - elapsed
    )


    cv2.putText(
        frame,
        f"Capturing: {remaining:.1f}s",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )


    cv2.imshow(
        "DeepGuard-X - Behaviour Capture",
        frame
    )


    # --------------------------------------------------
    # STOP AFTER REQUIRED DURATION
    # --------------------------------------------------

    if elapsed >= VIDEO_DURATION:

        break


    # --------------------------------------------------
    # PRESS Q TO STOP EARLY
    # --------------------------------------------------

    if cv2.waitKey(1) & 0xFF == ord("q"):

        print(
            "\nCapture stopped manually."
        )

        break


# ==================================================
# RELEASE CAMERA
# ==================================================

cap.release()

cv2.destroyAllWindows()


# ==================================================
# PHASE 2 — CAPTURE SUMMARY
# ==================================================

total_captured = len(
    captured_frames
)


print("\n======================================")
print("       CAPTURE COMPLETED")
print("======================================")

print(
    f"Behaviour Session     : {SESSION_ID}"
)

print(
    f"Total captured frames : {total_captured}"
)


# ==================================================
# CHECK CAPTURE
# ==================================================

if total_captured == 0:

    print(
        "ERROR: No frames captured."
    )

    exit()


# ==================================================
# PHASE 3 — SAVE ALL N FRAMES
#              FOR BEHAVIOUR
# ==================================================

print("\nSaving ALL captured frames")
print("for Behaviour Analysis...")


for i, frame in enumerate(
    captured_frames,
    start=1
):

    filename = os.path.join(
        BEHAVIOUR_FRAMES_FOLDER,
        f"frame_{i:04d}.jpg"
    )


    cv2.imwrite(
        filename,
        frame
    )


print(
    f"Behaviour frames saved : {total_captured}"
)

print(
    f"Behaviour location     : "
    f"{BEHAVIOUR_FRAMES_FOLDER}"
)


# ==================================================
# PHASE 4 — SELECT 25 COMMON FRAMES
#              FOR FACE ID / PAD / DEEPFAKE
# ==================================================

frames_to_select = min(
    TOTAL_FRAMES,
    total_captured
)


# --------------------------------------------------
# Select frames evenly across the complete
# captured sequence.
# --------------------------------------------------

selected_indices = np.linspace(
    0,
    total_captured - 1,
    frames_to_select,
    dtype=int
)


selected_frames = [

    captured_frames[i]

    for i in selected_indices

]


# ==================================================
# PHASE 5 — SAVE SELECTED FRAMES
# ==================================================

print("\n======================================")
print("       SELECTING COMMON FRAMES")
print("======================================")

print(
    f"Frames selected : {len(selected_frames)}"
)

print("\nSaving selected frames...")


for i, frame in enumerate(
    selected_frames,
    start=1
):

    filename = os.path.join(
        SELECTED_FOLDER,
        f"frame_{i:03d}.jpg"
    )


    cv2.imwrite(
        filename,
        frame
    )


# ==================================================
# PHASE 6 — FINAL SUMMARY
# ==================================================

print("\n======================================")
print("          CAPTURE READY")
print("======================================")

print(
    f"Behaviour Session : {SESSION_ID}"
)

print(
    f"Total captured    : {total_captured}"
)

print(
    f"Behaviour frames  : {total_captured}"
)

print(
    f"Selected frames   : {len(selected_frames)}"
)

print(
    f"Behaviour data    : "
    f"{BEHAVIOUR_FRAMES_FOLDER}"
)

print(
    f"25-frame data     : "
    f"{SELECTED_FOLDER}"
)

print("======================================")


# ==================================================
# FRAME USAGE
# ==================================================

print("\nFrame usage:")

print(
    "Face Identification → 25 selected frames"
)

print(
    "PAD                 → 25 selected frames"
)

print(
    "Deepfake Detection  → 25 selected frames"
)

print(
    "Behaviour Analysis  → ALL captured frames"
)

print("======================================")