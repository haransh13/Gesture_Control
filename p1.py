import cv2
import mediapipe as mp
from math import hypot
import screen_brightness_control as sbc
import numpy as np
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume  # For volume control
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL

# Initialize MediaPipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
    max_num_hands=2  # Allow tracking of both hands
)
draw = mp.solutions.drawing_utils

# Initialize volume control using Pycaw
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
minVol, maxVol, _ = volume.GetVolumeRange()

# Webcam setup
cap = cv2.VideoCapture(0)

# Define gesture thresholds
BRIGHTNESS_MIN_DIST, BRIGHTNESS_MAX_DIST = 15, 220
VOLUME_MIN_DIST, VOLUME_MAX_DIST = 15, 220

# Main program loop
while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)  # Flip frame horizontally
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
    results = hands.process(frameRGB)  # Detect hands
    landmarkLists = []  # Store landmarks for both hands

    # Process detected hands
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            landmarks = []
            h, w, _ = frame.shape

            # Store landmark positions
            for id, lm in enumerate(handLms.landmark):
                x, y = int(lm.x * w), int(lm.y * h)
                landmarks.append([id, x, y])

            landmarkLists.append(landmarks)
            draw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)  # Draw hand landmarks

    # Initialize hand data
    leftHand, rightHand = None, None

    # Assign hands based on location
    if len(landmarkLists) == 1:
        hand = landmarkLists[0]
        if hand[0][1] < frame.shape[1] // 2:  # Left side of the screen
            leftHand = hand
        else:
            rightHand = hand
    elif len(landmarkLists) == 2:
        if landmarkLists[0][0][1] < landmarkLists[1][0][1]:
            leftHand, rightHand = landmarkLists[0], landmarkLists[1]
        else:
            leftHand, rightHand = landmarkLists[1], landmarkLists[0]

    # Control brightness with the right hand
    if rightHand:
        x1, y1 = rightHand[4][1], rightHand[4][2]  # Thumb tip
        x2, y2 = rightHand[8][1], rightHand[8][2]  # Index tip
        cv2.circle(frame, (x1, y1), 7, (0, 255, 0), cv2.FILLED)
        cv2.circle(frame, (x2, y2), 7, (0, 255, 0), cv2.FILLED)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        brightness_distance = hypot(x2 - x1, y2 - y1)
        brightness_level = np.interp(brightness_distance, [BRIGHTNESS_MIN_DIST, BRIGHTNESS_MAX_DIST], [0, 100])
        sbc.set_brightness(int(brightness_level))

    # Control volume with the left hand
    if leftHand:
        x1, y1 = leftHand[4][1], leftHand[4][2]  # Thumb tip
        x2, y2 = leftHand[8][1], leftHand[8][2]  # Index tip
        cv2.circle(frame, (x1, y1), 7, (255, 0, 0), cv2.FILLED)
        cv2.circle(frame, (x2, y2), 7, (255, 0, 0), cv2.FILLED)
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)

        volume_distance = hypot(x2 - x1, y2 - y1)
        volume_level = np.interp(volume_distance, [VOLUME_MIN_DIST, VOLUME_MAX_DIST], [minVol, maxVol])
        volume.SetMasterVolumeLevel(volume_level, None)

    # Display the frame
    cv2.imshow("Hand Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Quit on 'q' key press
        break

cap.release()
cv2.destroyAllWindows()
