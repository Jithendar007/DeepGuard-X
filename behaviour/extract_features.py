import sys
import cv2
import os
import csv
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp


# ============================================================
# SETTINGS
# ============================================================

BEHAVIOUR_ROOT_FOLDER = "experiments/behaviour_sessions"
EXPERIMENTS_ROOT_FOLDER = "experiments"

MODEL_PATH = (
    "behaviour/models/face_landmarker.task"
)


# ============================================================
# FIND SESSION
# ============================================================

if len(sys.argv) > 1:

    SESSION_ID = sys.argv[1].upper()

    is_behaviour_session = (
        SESSION_ID.startswith("B")
        and SESSION_ID[1:].isdigit()
    )

    is_legacy_session = (
        SESSION_ID.startswith("V")
        and SESSION_ID[1:].isdigit()
    )

    if not (is_behaviour_session or is_legacy_session):

        print(
            "ERROR: Invalid session ID."
        )

        print(
            "Examples:"
        )

        print(
            "  python behaviour\\extract_features.py B002"
        )

        print(
            "  python behaviour\\extract_features.py V001"
        )

        exit()

    if is_behaviour_session:

        ROOT_FOLDER = BEHAVIOUR_ROOT_FOLDER

    else:

        ROOT_FOLDER = EXPERIMENTS_ROOT_FOLDER

else:

    # If no session is provided,
    # automatically use the latest session.

    if not os.path.exists(BEHAVIOUR_ROOT_FOLDER):

        print("ERROR: Behaviour sessions folder not found.")
        exit()

    sessions = []

    for name in os.listdir(BEHAVIOUR_ROOT_FOLDER):

        if name.startswith("B") and name[1:].isdigit():

            sessions.append(
                int(name[1:])
            )

    if not sessions:

        print("ERROR: No behaviour sessions found.")
        exit()

    latest_session_number = max(sessions)

    SESSION_ID = f"B{latest_session_number:03d}"

    ROOT_FOLDER = BEHAVIOUR_ROOT_FOLDER

SESSION_FOLDER = os.path.join(
    ROOT_FOLDER,
    SESSION_ID
)

FRAMES_FOLDER = os.path.join(
    SESSION_FOLDER,
    "behaviour_frames"
)

OUTPUT_FILE = os.path.join(
    SESSION_FOLDER,
    "behaviour_sequence.csv"
)


# ============================================================
# CHECK MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):

    print("ERROR: Face Landmarker model not found.")

    print(
        f"Expected: {MODEL_PATH}"
    )

    exit()


# ============================================================
# CHECK FRAME FOLDER
# ============================================================

if not os.path.exists(FRAMES_FOLDER):

    print("ERROR: Behaviour frames folder not found.")

    print(
        f"Expected: {FRAMES_FOLDER}"
    )

    exit()


# ============================================================
# GET FRAME FILES
# ============================================================

frame_files = [

    file
    for file in os.listdir(FRAMES_FOLDER)

    if file.lower().endswith(
        (".jpg", ".jpeg", ".png")
    )

]


# Sort using frame number

def frame_number(filename):

    digits = "".join(
        character
        for character in filename
        if character.isdigit()
    )

    if digits:
        return int(digits)

    return 0


frame_files.sort(
    key=frame_number
)


if not frame_files:

    print("ERROR: No behaviour frames found.")
    exit()


# ============================================================
# CREATE MEDIAPIPE FACE LANDMARKER
# ============================================================

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.FaceLandmarkerOptions(

    base_options=base_options,

    running_mode=(
        vision.RunningMode.IMAGE
    ),

    num_faces=1,

    min_face_detection_confidence=0.5,

    min_face_presence_confidence=0.5,

    min_tracking_confidence=0.5,

    output_face_blendshapes=False,

    output_facial_transformation_matrixes=False

)


# ============================================================
# LANDMARK INDICES
# ============================================================

LEFT_EYE = [
    33,
    160,
    158,
    133,
    153,
    144
]

RIGHT_EYE = [
    362,
    385,
    387,
    263,
    373,
    380
]


# ============================================================
# DISTANCE
# ============================================================

def distance(point1, point2):

    return np.linalg.norm(
        np.array(point1)
        -
        np.array(point2)
    )


# ============================================================
# EAR
# ============================================================

def calculate_ear(
    landmarks,
    eye
):

    p1 = landmarks[eye[0]]
    p2 = landmarks[eye[1]]
    p3 = landmarks[eye[2]]
    p4 = landmarks[eye[3]]
    p5 = landmarks[eye[4]]
    p6 = landmarks[eye[5]]


    vertical_1 = distance(
        p2,
        p6
    )

    vertical_2 = distance(
        p3,
        p5
    )

    horizontal = distance(
        p1,
        p4
    )


    if horizontal == 0:

        return 0.0


    return (
        vertical_1 +
        vertical_2
    ) / (
        2.0 *
        horizontal
    )


# ============================================================
# MAR
# ============================================================

def calculate_mar(
    landmarks
):

    left = landmarks[61]
    right = landmarks[291]

    top = landmarks[13]
    bottom = landmarks[14]


    mouth_width = distance(
        left,
        right
    )

    mouth_height = distance(
        top,
        bottom
    )


    if mouth_width == 0:

        return 0.0


    return (
        mouth_height /
        mouth_width
    )


# ============================================================
# HEAD POSE
# ============================================================

def calculate_head_pose(
    landmarks,
    image_width,
    image_height
):

    image_points = np.array([

        landmarks[1],

        landmarks[152],

        landmarks[33],

        landmarks[263],

        landmarks[61],

        landmarks[291]

    ], dtype=np.float64)


    model_points = np.array([

        (0.0, 0.0, 0.0),

        (0.0, -63.6, -12.5),

        (-43.3, 32.7, -26.0),

        (43.3, 32.7, -26.0),

        (-28.9, -28.9, -24.1),

        (28.9, -28.9, -24.1)

    ], dtype=np.float64)


    focal_length = image_width

    center = (

        image_width / 2.0,

        image_height / 2.0

    )


    camera_matrix = np.array([

        [
            focal_length,
            0,
            center[0]
        ],

        [
            0,
            focal_length,
            center[1]
        ],

        [
            0,
            0,
            1
        ]

    ], dtype=np.float64)


    distortion = np.zeros(
        (4, 1)
    )


    success, rotation_vector, translation_vector = cv2.solvePnP(

        model_points,

        image_points,

        camera_matrix,

        distortion,

        flags=cv2.SOLVEPNP_ITERATIVE

    )


    if not success:

        return (
            0.0,
            0.0,
            0.0
        )


    rotation_matrix, _ = cv2.Rodrigues(
        rotation_vector
    )


    projection_matrix = np.hstack(
        (
            rotation_matrix,
            translation_vector
        )
    )


    _, _, _, _, _, _, euler_angles = (
        cv2.decomposeProjectionMatrix(
            projection_matrix
        )
    )


    pitch = float(
        euler_angles[0]
    )

    yaw = float(
        euler_angles[1]
    )

    roll = float(
        euler_angles[2]
    )


    return (
        yaw,
        pitch,
        roll
    )


# ============================================================
# START EXTRACTION
# ============================================================

print("\n======================================")
print("      TEMPORAL FEATURE EXTRACTION")
print("======================================")

print(
    f"Session : {SESSION_ID}"
)

print(
    f"Frames  : {len(frame_files)}"
)

print(
    f"Model   : {MODEL_PATH}"
)

print("\nProcessing frames...\n")


sequence = []

previous_values = None


# ============================================================
# RUN FACE LANDMARKER
# ============================================================

with vision.FaceLandmarker.create_from_options(
    options
) as landmarker:

    for index, filename in enumerate(
        frame_files,
        start=1
    ):

        path = os.path.join(
            FRAMES_FOLDER,
            filename
        )

        image = cv2.imread(path)

        if image is None:
            print(
                f"Frame {index}: could not read."
            )
            continue

        image_height, image_width = image.shape[:2]

        mp_image = mp.Image.create_from_file(
            path
        )

        result = landmarker.detect(
            mp_image
        )

        if not result.face_landmarks:
            print(
                f"Frame {index}: face not detected."
            )
            continue

        face_landmarks = result.face_landmarks[0]

        # ====================================================
        # CONVERT LANDMARKS TO PIXEL COORDINATES
        # ====================================================

        landmarks = [
            (
                landmark.x * image_width,
                landmark.y * image_height
            )
            for landmark in face_landmarks
        ]

        # ====================================================
        # BASIC FEATURES
        # ====================================================

        left_ear = calculate_ear(
            landmarks,
            LEFT_EYE
        )

        right_ear = calculate_ear(
            landmarks,
            RIGHT_EYE
        )

        ear = (
            left_ear +
            right_ear
        ) / 2.0

        mar = calculate_mar(
            landmarks
        )

        yaw, pitch, roll = calculate_head_pose(
            landmarks,
            image_width,
            image_height
        )

        # ====================================================
        # TEMPORAL DIFFERENCES
        # ====================================================

        if previous_values is None:

            d_ear = 0.0
            d_mar = 0.0
            d_yaw = 0.0
            d_pitch = 0.0
            d_roll = 0.0

        else:

            d_ear = abs(
                ear -
                previous_values["ear"]
            )

            d_mar = abs(
                mar -
                previous_values["mar"]
            )

            d_yaw = abs(
                yaw -
                previous_values["yaw"]
            )

            d_pitch = abs(
                pitch -
                previous_values["pitch"]
            )

            d_roll = abs(
                roll -
                previous_values["roll"]
            )

        # ====================================================
        # STORE FRAME FEATURES
        # ====================================================

        row = {
            "frame": index,

            "ear": ear,
            "mar": mar,

            "yaw": yaw,
            "pitch": pitch,
            "roll": roll,

            "d_ear": d_ear,
            "d_mar": d_mar,

            "d_yaw": d_yaw,
            "d_pitch": d_pitch,
            "d_roll": d_roll
        }

        sequence.append(
            row
        )

        # ====================================================
        # SAVE CURRENT VALUES FOR NEXT FRAME
        # ====================================================

        previous_values = {
            "ear": ear,
            "mar": mar,

            "yaw": yaw,
            "pitch": pitch,
            "roll": roll
        }

        print(
            f"Frame {index:03d} → "
            f"EAR: {ear:.4f} | "
            f"MAR: {mar:.4f} | "
            f"Yaw: {yaw:7.2f}° | "
            f"Pitch: {pitch:7.2f}° | "
            f"Roll: {roll:7.2f}°"
        )


# ============================================================
# CHECK RESULT
# ============================================================

if not sequence:

    print(
        "\nERROR: No valid feature rows generated."
    )

    exit()


# ============================================================
# SAVE TEMPORAL CSV
# ============================================================

fieldnames = [

    "frame",

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


with open(
    OUTPUT_FILE,
    "w",
    newline=""
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(
        sequence
    )


# ============================================================
# FINAL
# ============================================================

print("\n======================================")
print("      EXTRACTION COMPLETED")
print("======================================")

print(
    f"Session             : {SESSION_ID}"
)

print(
    f"Frames found        : {len(frame_files)}"
)

print(
    f"Frames processed    : {len(sequence)}"
)

print(
    f"Temporal features   : {len(fieldnames) - 1}"
)

print(
    f"CSV file            : {OUTPUT_FILE}"
)

print("======================================")