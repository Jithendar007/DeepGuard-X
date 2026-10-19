import cv2
import insightface
import numpy as np
import os

# ==================================================
# LOAD INSIGHTFACE
# ==================================================

face_app = insightface.app.FaceAnalysis(
    name="buffalo_l"
)

face_app.prepare(
    ctx_id=-1,
    det_size=(320, 320)
)

THRESHOLD = 0.5
# ==================================================
# LOAD REGISTERED PEOPLE
# ==================================================

DATA_FOLDER = "data"
registered_people = {}

for person_name in os.listdir(DATA_FOLDER):

    person_folder = os.path.join(
        DATA_FOLDER,
        person_name
    )

    embedding_path = os.path.join(
        person_folder,
        "embedding.npy"
    )

    if os.path.isfile(embedding_path):

        embedding = np.load(embedding_path)

        embedding = (
            embedding /
            np.linalg.norm(embedding)
        )

        registered_people[person_name] = embedding


if len(registered_people) == 0:

    print("No registered users found.")
    exit()


# ==================================================
# LOAD COMMON FRAMES
# ==================================================

FRAME_FOLDER = "experiments/V001"

frame_files = sorted(
    [
        f for f in os.listdir(FRAME_FOLDER)
        if f.endswith(".jpg")
    ]
)

print("\n======================================")
print("       IDENTITY PROCESSING")
print("======================================")

print(
    f"Common frames found : {len(frame_files)}"
)


# ==================================================
# STORE SCORES
# ==================================================

person_scores = {
    person_name: []
    for person_name in registered_people
}


# ==================================================
# PROCESS EVERY COMMON FRAME
# ==================================================

for i, filename in enumerate(
    frame_files,
    start=1
):

    image_path = os.path.join(
        FRAME_FOLDER,
        filename
    )

    frame = cv2.imread(image_path)

    if frame is None:

        print(
            f"Frame {i:02d}: Could not read"
        )

        continue


    # ----------------------------------------------
    # Detect face
    # ----------------------------------------------

    faces = face_app.get(frame)

    if len(faces) == 0:

        print(
            f"Frame {i:02d}: No face detected"
        )

        continue


    # ----------------------------------------------
    # Select largest face
    # ----------------------------------------------

    face = max(
        faces,
        key=lambda f:
        (f.bbox[2] - f.bbox[0]) *
        (f.bbox[3] - f.bbox[1])
    )


    # ----------------------------------------------
    # Current embedding
    # ----------------------------------------------

    current_embedding = face.embedding

    current_embedding = (
        current_embedding /
        np.linalg.norm(current_embedding)
    )


    # ----------------------------------------------
    # Compare with registered people
    # ----------------------------------------------

    frame_best_name = None
    frame_best_score = -1

    for person_name, stored_embedding in registered_people.items():

        similarity = np.dot(
            stored_embedding,
            current_embedding
        )

        person_scores[person_name].append(
            float(similarity)
        )

        if similarity > frame_best_score:

            frame_best_score = similarity
            frame_best_name = person_name


    print(
        f"Frame {i:02d} → "
        f"{frame_best_name}: "
        f"{frame_best_score:.4f}"
    )


# ==================================================
# AGGREGATION
# ==================================================

valid_people = {}

for person_name, scores in person_scores.items():

    if len(scores) > 0:

        valid_people[person_name] = np.mean(
            scores
        )


# ==================================================
# FINAL RESULT
# ==================================================

print("\n======================================")
print("       IDENTITY AGGREGATION")
print("======================================")

if len(valid_people) == 0:

    print("No valid face frames.")
    exit()


# Find the enrolled person with the highest
# aggregated similarity score
best_name = max(
    valid_people,
    key=valid_people.get
)

final_identity_score = valid_people[best_name]

# Get frame scores of the highest-scoring person
best_scores = np.array(
    person_scores[best_name]
)


print(
    f"Highest scoring person : {best_name}"
)

print(
    f"Values used            : {len(best_scores)}"
)

print(
    "Frame scores           :",
    np.round(best_scores, 4)
)

print("--------------------------------------")

print(
    f"Aggregated score       : "
    f"{final_identity_score:.4f}"
)

print(
    f"Percentage             : "
    f"{final_identity_score * 100:.2f}%"
)

print(
    f"Threshold              : "
    f"{THRESHOLD:.4f}"
)

print("--------------------------------------")


# ------------------------------------------
# FINAL IDENTITY DECISION
# ------------------------------------------

if final_identity_score >= THRESHOLD:

    print(
        f"FINAL IDENTITY        : {best_name}"
    )

    print(
        "FINAL DECISION        : MATCH"
    )

else:

    print(
        "FINAL IDENTITY        : UNKNOWN"
    )

    print(
        "FINAL DECISION        : REJECT"
    )


print("======================================")