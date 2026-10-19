import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "behaviour/models/face_landmarker.task"


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
        np.array(point1) -
        np.array(point2)
    )


# ============================================================
# EAR
# ============================================================

def calculate_ear(landmarks, eye):

    p1 = landmarks[eye[0]]
    p2 = landmarks[eye[1]]
    p3 = landmarks[eye[2]]
    p4 = landmarks[eye[3]]
    p5 = landmarks[eye[4]]
    p6 = landmarks[eye[5]]

    vertical_1 = distance(p2, p6)

    vertical_2 = distance(p3, p5)

    horizontal = distance(p1, p4)

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

def calculate_mar(landmarks):

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
# REAL-TIME BEHAVIOUR FEATURE EXTRACTOR
# ============================================================

class RealtimeBehaviourExtractor:

    def __init__(self):

        print("\nLoading MediaPipe Face Landmarker...")

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

        self.landmarker = (
            vision.FaceLandmarker
            .create_from_options(options)
        )

        self.sequence = []

        self.previous_values = None

        self.frame_number = 0

    # ========================================================
    # PROCESS ONE CAMERA FRAME
    # ========================================================

    def process_frame(self, frame):

        self.frame_number += 1

        image_height, image_width = (
            frame.shape[:2]
        )

        # ----------------------------------------------------
        # OpenCV BGR → RGB
        # ----------------------------------------------------

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # ----------------------------------------------------
        # MediaPipe detection
        # ----------------------------------------------------

        result = self.landmarker.detect(
            mp_image
        )

        if not result.face_landmarks:

            return False

        face_landmarks = result.face_landmarks[0]

        # ----------------------------------------------------
        # Convert landmarks to pixels
        # ----------------------------------------------------

        landmarks = [

            (
                landmark.x * image_width,
                landmark.y * image_height
            )

            for landmark in face_landmarks
        ]

        # ----------------------------------------------------
        # Basic features
        # ----------------------------------------------------

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

        yaw, pitch, roll = (
            calculate_head_pose(
                landmarks,
                image_width,
                image_height
            )
        )

        # ----------------------------------------------------
        # Temporal differences
        # ----------------------------------------------------

        if self.previous_values is None:

            d_ear = 0.0
            d_mar = 0.0
            d_yaw = 0.0
            d_pitch = 0.0
            d_roll = 0.0

        else:

            d_ear = abs(
                ear -
                self.previous_values["ear"]
            )

            d_mar = abs(
                mar -
                self.previous_values["mar"]
            )

            d_yaw = abs(
                yaw -
                self.previous_values["yaw"]
            )

            d_pitch = abs(
                pitch -
                self.previous_values["pitch"]
            )

            d_roll = abs(
                roll -
                self.previous_values["roll"]
            )

        # ----------------------------------------------------
        # Store row
        # ----------------------------------------------------

        row = {

            "frame": self.frame_number,

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

        self.sequence.append(row)

        # ----------------------------------------------------
        # Previous values
        # ----------------------------------------------------

        self.previous_values = {

            "ear": ear,

            "mar": mar,

            "yaw": yaw,

            "pitch": pitch,

            "roll": roll

        }

        return True

    # ========================================================
    # GET SEQUENCE
    # ========================================================

    def get_sequence(self):

        return self.sequence

    # ========================================================
    # CLOSE
    # ========================================================

    def close(self):

        self.landmarker.close()
    