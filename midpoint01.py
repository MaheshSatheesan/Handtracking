import cv2
import mediapipe as mp
import time

# Initialize webcam (camera index 0)
cap = cv2.VideoCapture(1)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# MediaPipe Hands configuration
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    # Read a frame from the webcam
    success, img = cap.read()

    # Check if the frame is empty
    if not success:
        print("Error: Failed to read frame from the webcam.")
        break

    # Convert BGR image to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process hand landmarks using MediaPipe
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

        # Check if two hands were detected before accessing landmarks
        if len(results.multi_hand_landmarks) == 2:
            x1, y1 = results.multi_hand_landmarks[0].landmark[4].x * w, results.multi_hand_landmarks[0].landmark[4].y * h
            x2, y2 = results.multi_hand_landmarks[1].landmark[4].x * w, results.multi_hand_landmarks[1].landmark[4].y * h

            # Calculate the midpoint between the two thumbs
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2

            # Draw a point at the midpoint
            cv2.circle(img, (int(mid_x), int(mid_y)), 5, (0, 0, 255), cv2.FILLED)

    # Calculate and display FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    # Display the processed frame
    cv2.imshow("Image", img)

    # Exit the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
