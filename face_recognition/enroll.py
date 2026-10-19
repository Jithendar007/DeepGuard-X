import cv2
import insightface
import numpy as np
import os
import time

# -----------------------------
# Load InsightFace
# -----------------------------
face_app = insightface.app.FaceAnalysis(
    name="buffalo_l"
)

face_app.prepare(
    ctx_id=-1,
    det_size=(320, 320)
)

# -----------------------------
# Ask for person's name
# -----------------------------
name = input("Enter person's name: ").strip()

if not name:
    print("Name cannot be empty.")
    exit()

# Remove characters that are unsafe in folder names
safe_name = "".join(
    c for c in name
    if c.isalnum() or c in (" ", "_", "-")
).strip()

if not safe_name:
    print("Invalid name.")
    exit()

# -----------------------------
# Create person's folder
# -----------------------------
person_folder = os.path.join(
    "data",
    safe_name
)

os.makedirs(person_folder, exist_ok=True)

# -----------------------------
# Guided poses
# -----------------------------
poses = [
    "Look STRAIGHT at the camera",
    "Turn your face SLIGHTLY LEFT",
    "Turn your face SLIGHTLY RIGHT",
    "Look SLIGHTLY UP",
    "Look SLIGHTLY DOWN"
]

embeddings = []

# -----------------------------
# Open webcam
# -----------------------------
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print()
print(f"Starting enrollment for: {name}")
print("Press Q at any time to cancel.")
print()

# Give camera time to start
time.sleep(1)

# -----------------------------
# Capture each pose
# -----------------------------
for pose_number, instruction in enumerate(poses):

    print(f"Pose {pose_number + 1}/5: {instruction}")

    # Give user time to position their face
    countdown = 3

    while countdown > 0:

        start_time = time.time()

        while time.time() - start_time < 1:

            success, frame = cap.read()

            if not success:
                continue

            cv2.putText(
                frame,
                instruction,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Get ready: {countdown}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            cv2.imshow(
                "DeepGuard-X - Enrollment",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                print("Enrollment cancelled.")
                exit()

        countdown -= 1

    # -------------------------
    # Find a good frame
    # -------------------------
    captured = False

    while not captured:

        success, frame = cap.read()

        if not success:
            continue

        faces = face_app.get(frame)

        if len(faces) == 1:

            face = faces[0]

            x1, y1, x2, y2 = face.bbox.astype(int)

            # Draw face box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Get face size
            face_width = x2 - x1
            face_height = y2 - y1

            # Require reasonable face size
            if face_width >= 100 and face_height >= 100:

                embeddings.append(
                    face.embedding.copy()
                )

                captured = True

                print(
                    f"✓ Pose {pose_number + 1}/5 captured"
                )

                cv2.putText(
                    frame,
                    "CAPTURED!",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

            else:

                cv2.putText(
                    frame,
                    "Move closer",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

        elif len(faces) == 0:

            cv2.putText(
                frame,
                "Face not detected",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        else:

            cv2.putText(
                frame,
                "Only one person allowed",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        cv2.putText(
            frame,
            instruction,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.imshow(
            "DeepGuard-X - Enrollment",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            print("Enrollment cancelled.")
            exit()

    # Short pause before next pose
    time.sleep(1)

# -----------------------------
# Close camera
# -----------------------------
cap.release()
cv2.destroyAllWindows()

# -----------------------------
# Create average embedding
# -----------------------------
embeddings = np.array(embeddings)

average_embedding = np.mean(
    embeddings,
    axis=0
)

# Normalize
average_embedding = (
    average_embedding /
    np.linalg.norm(average_embedding)
)

# -----------------------------
# Save embedding
# -----------------------------
embedding_path = os.path.join(
    person_folder,
    "embedding.npy"
)

np.save(
    embedding_path,
    average_embedding
)

print()
print("================================")
print("Enrollment successful!")
print("================================")
print(f"Name: {name}")
print(f"Samples: {len(embeddings)}")
print(f"Embedding shape: {average_embedding.shape}")
print(f"Saved to: {embedding_path}")
print()