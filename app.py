import cv2
import dlib
import numpy as np
from scipy.spatial import distance
import os

# -----------------------------
# Configuration
# -----------------------------
EAR_THRESHOLD = 0.25
MODEL_PATH = "models/shape_predictor_68_face_landmarks.dat"

# -----------------------------
# Function to calculate EAR
# -----------------------------
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# -----------------------------
# Check if model file exists
# -----------------------------
if not os.path.exists(MODEL_PATH):
    print("Error: shape_predictor_68_face_landmarks.dat not found.")
    print("Please download the model and place it inside the models/ folder.")
    exit()

# -----------------------------
# Initialize Dlib Detector
# -----------------------------
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(MODEL_PATH)

# Eye landmark indexes
(lStart, lEnd) = (42, 48)
(rStart, rEnd) = (36, 42)

# -----------------------------
# Start Webcam
# -----------------------------
cap = cv2.VideoCapture(0)

print("Starting Drowsiness Detection... Press ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)
        shape = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        ear = (leftEAR + rightEAR) / 2.0

        # Draw eye contours
        cv2.polylines(frame, [leftEye], True, (0, 255, 0), 1)
        cv2.polylines(frame, [rightEye], True, (0, 255, 0), 1)

        # Display EAR value
        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Check for drowsiness
        if ear < EAR_THRESHOLD:
            cv2.putText(frame, "DROWSINESS ALERT!",
                        (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 0, 255),
                        3)

    cv2.imshow("Drowsiness Detection", frame)

    # Exit on ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
