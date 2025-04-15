#! /usr/bin/env python3.11
import cv2 as cv
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


# Initialize webcam
cap = cv.VideoCapture(0)
def get_gesture(landmarks):
    thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y
    thumb_ip = landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y
    index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    index_pip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
    middle_tip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
    middle_pip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
    ring_tip = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y
    ring_pip = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y
    pinky_tip = landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y
    pinky_pip = landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y

    # Check for ASL 'A' (Thumb extended, other fingers bent)
    thumb_extended = thumb_tip < thumb_ip
    fingers_bent = (index_tip > index_pip and 
                    middle_tip > middle_pip and 
                    ring_tip > ring_pip and 
                    pinky_tip > pinky_pip)
    if thumb_extended and fingers_bent:
        return "A"
    
    # Check for ASL 'B' (Four fingers extended, thumb bent)
    fingers_extended = (index_tip < index_pip and 
                        middle_tip < middle_pip and 
                        ring_tip < ring_pip and 
                        pinky_tip < pinky_pip)
    thumb_bent = thumb_tip > thumb_ip
    if fingers_extended and thumb_bent:
        return "B"

    if thumb_tip < min(index_tip,middle_tip,ring_tip,pinky_tip):
        return "Thumbs Up"
    elif index_tip < min(thumb_tip, ring_tip, pinky_tip) and middle_tip < min(thumb_tip, ring_tip, pinky_tip):
        return "Peace"
    else:
        return "Unknown"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue
    flipped_frame = cv.flip(frame,1)
    # Convert the BGR image to RGB
    rgb_frame = cv.cvtColor(flipped_frame, cv.COLOR_BGR2RGB)

    # Process the frame and detect hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(flipped_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get hand gesture
            gesture = get_gesture(hand_landmarks)
            cv.putText(flipped_frame, f"Gesture: {gesture}", (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            height, width, _ = flipped_frame.shape
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_finger_tip_x = int(index_finger_tip.x * width)
            index_finger_tip_y = int(index_finger_tip.y * height)
            cv.circle(flipped_frame, (index_finger_tip_x, index_finger_tip_y), 5, (100, 255, 100), -1)
    
    cv.imshow('Hand Gesture Detection', flipped_frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
