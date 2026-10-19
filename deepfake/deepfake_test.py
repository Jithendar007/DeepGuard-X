import os

import cv2

try:
    from .inference import process_frames
except ImportError:
    from inference import process_frames


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAME_FOLDER = os.path.join(BASE_DIR, "experiments", "V001", "selected_frames")


def main():
    if not os.path.isdir(FRAME_FOLDER):
        print(f"ERROR: Frame folder not found: {FRAME_FOLDER}")
        return 1

    frame_files = sorted(
        filename
        for filename in os.listdir(FRAME_FOLDER)
        if filename.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    if not frame_files:
        print("ERROR: No frames found.")
        return 1

    frames = []
    for filename in frame_files:
        image = cv2.imread(os.path.join(FRAME_FOLDER, filename))
        if image is not None:
            frames.append(image)

    print("\n======================================")
    print("       DEEPFAKE PROCESSING")
    print("======================================")
    print(f"Common frames found : {len(frame_files)}")

    score = process_frames(frames)

    print("\n======================================")
    print("       DEEPFAKE RESULT")
    print("======================================")
    print(f"Probability of Real            : {score * 100:.2f}%")
    print("======================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
