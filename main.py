#! /usr/bin/env python3.11
import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)
def get_gesture(landmarks):
    # Implement your gesture recognition logic here
    # This is a simple example - you can expand on this
    thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y
    index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    middle_tip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y

    if thumb_tip < index_tip and thumb_tip < middle_tip:
        return "Thumbs Up"
    elif index_tip < thumb_tip and middle_tip < thumb_tip:
        return "Peace"
    else:
        return "Unknown"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get hand gesture
            gesture = get_gesture(hand_landmarks)
            cv2.putText(frame, f"Gesture: {gesture}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Hand Gesture Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()