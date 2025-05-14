import cv2
import mediapipe as mp
import numpy as np

# Initialize the MediaPipe hands module for hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)  # Detect only one hand
mp_drawing = mp.solutions.drawing_utils  # For drawing hand landmarks

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # Read frame from the webcam
    if not ret:
        break

    # Flip the frame horizontally for a more intuitive display
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB because MediaPipe uses RGB images
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame for hand landmarks detection
    results = hands.process(rgb_frame)

    # If a hand is detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmark points
            landmarks = hand_landmarks.landmark
            h, w, _ = frame.shape

            # Finger tip landmarks (tip of thumb, index, middle, ring, pinky fingers)
            fingertips = [4, 8, 12, 16, 20]

            # Track the number of fingers that are raised (extended)
            finger_count = 0

            for i in fingertips:
                # Get the y-coordinate of the tip and the lower knuckle
                tip_y = landmarks[i].y * h
                lower_y = landmarks[i - 2].y * h  # Compare with the point two below the tip

                # If the tip is above the knuckle, count as extended
                if tip_y < lower_y:
                    finger_count += 1

            # Display the number of fingers detected on the frame
            cv2.putText(frame, f'Fingers: {finger_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Hand Gesture Recognition", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
