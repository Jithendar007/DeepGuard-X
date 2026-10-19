import cv2
import os
import time
import shutil

# ============================================================
# SETTINGS
# ============================================================

VIDEO_DURATION = 5

SELECTED_FPS = 5       # 25 frames approximately
BEHAVIOUR_FPS = 30     # 150 frames approximately

BASE_FOLDER = "experiments/V001"

SELECTED_FOLDER = os.path.join(
    BASE_FOLDER,
    "selected_frames"
)

BEHAVIOUR_FOLDER = os.path.join(
    BASE_FOLDER,
    "behaviour_frames"
)


# ============================================================
# PREPARE FOLDERS
# ============================================================

if os.path.exists(SELECTED_FOLDER):
    shutil.rmtree(SELECTED_FOLDER)

if os.path.exists(BEHAVIOUR_FOLDER):
    shutil.rmtree(BEHAVIOUR_FOLDER)

os.makedirs(SELECTED_FOLDER, exist_ok=True)
os.makedirs(BEHAVIOUR_FOLDER, exist_ok=True)


# ============================================================
# OPEN CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()


print("\n======================================")
print("       DEEPGUARD-X COMMON CAPTURE")
print("======================================")
print("Duration : 5 seconds")
print("Selected : approximately 25 frames")
print("Behaviour: approximately 150 frames")
print("\nLook at the camera...")
print("Capture starting...\n")


# ============================================================
# CAPTURE
# ============================================================

start_time = time.time()

last_selected_time = -1
last_behaviour_time = -1

selected_count = 0
behaviour_count = 0


while True:

    success, frame = cap.read()

    if not success:
        print("ERROR: Could not read camera.")
        break

    current_time = time.time()
    elapsed = current_time - start_time

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

    # ========================================================
    # 150 BEHAVIOUR FRAMES
    # ========================================================

    behaviour_interval = 1.0 / BEHAVIOUR_FPS

    if (
        current_time - last_behaviour_time
        >= behaviour_interval
    ):

        last_behaviour_time = current_time

        behaviour_count += 1

        filename = os.path.join(
            BEHAVIOUR_FOLDER,
            f"frame_{behaviour_count:03d}.jpg"
        )

        cv2.imwrite(
            filename,
            frame
        )


    # ========================================================
    # 25 SELECTED FRAMES
    # ========================================================

    selected_interval = 1.0 / SELECTED_FPS

    if (
        current_time - last_selected_time
        >= selected_interval
    ):

        last_selected_time = current_time

        selected_count += 1

        filename = os.path.join(
            SELECTED_FOLDER,
            f"frame_{selected_count:03d}.jpg"
        )

        cv2.imwrite(
            filename,
            frame
        )


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "DeepGuard-X - Common Capture",
        frame
    )


    # ========================================================
    # STOP
    # ========================================================

    if elapsed >= VIDEO_DURATION:
        break

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()


# ============================================================
# RESULT
# ============================================================

print("\n======================================")
print("          CAPTURE COMPLETED")
print("======================================")

print(
    f"Selected frames  : {selected_count}"
)

print(
    f"Behaviour frames : {behaviour_count}"
)

print(
    f"Selected folder  : {SELECTED_FOLDER}"
)

print(
    f"Behaviour folder : {BEHAVIOUR_FOLDER}"
)

print("======================================")