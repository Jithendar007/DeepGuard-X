import cv2
import mediapipe as mp

# MediaPipe Face Detector
BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Face detector configuration
options = FaceDetectorOptions(
    base_options=BaseOptions(
        model_asset_path="models/face_detector.tflite"
    ),
    running_mode=VisionRunningMode.IMAGE,
    min_detection_confidence=0.5
)

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Camera started. Press Q to quit.")

# Create detector
with FaceDetector.create_from_options(options) as detector:

    while True:

        # Read frame
        success, frame = cap.read()

        if not success:
            print("Error: Could not read camera frame.")
            break

        # Convert BGR → RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Detect face
        result = detector.detect(mp_image)

        # Draw detected faces
        if result.detections:

            for detection in result.detections:

                bbox = detection.bounding_box

                x = bbox.origin_x
                y = bbox.origin_y
                width = bbox.width
                height = bbox.height

                # Draw rectangle
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + width, y + height),
                    (0, 255, 0),
                    2
                )

                # Display confidence
                score = detection.categories[0].score

                cv2.putText(
                    frame,
                    f"Face: {score:.2f}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

        # Show camera
        cv2.imshow("DeepGuard-X - Face Detection", frame)

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Release camera
cap.release()
cv2.destroyAllWindows()