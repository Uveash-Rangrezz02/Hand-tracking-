import cv2
import imutils
import numpy as np
import mediapipe as mp

# Mediapipe Hand Tracking Setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Initialize game variables
score = 0
max_score = 20
list_capacity = 0
max_lc = 20
l = []
flag = 0
apple_x = None
apple_y = None
center = None
prev_c = None

# Distance function
def dist(pt1, pt2):
    return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

# Capture video from webcam
cap = cv2.VideoCapture(0)

# Game loop
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    img = imutils.resize(frame.copy(), width=600)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Hand Detection
    result = hands.process(img_rgb)
    
    if apple_x is None or apple_y is None:
        apple_x = np.random.randint(30, frame.shape[1] - 30)
        apple_y = np.random.randint(100, 350)

    cv2.circle(frame, (apple_x, apple_y), 5, (0, 0, 255), -1)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get Index Finger Tip Position
            index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            h, w, c = frame.shape
            center = (int(index_finger.x * w), int(index_finger.y * h))

            cv2.circle(frame, center, 10, (255, 0, 0), -1)

            if len(l) > list_capacity:
                l = l[1:]

            if prev_c and center and dist(prev_c, center) > 3.5:
                l.append(center)

            apple = (apple_x, apple_y)
            if center and dist(apple, center) < 10:
                score += 1
                if score == max_score:
                    flag = 1
                list_capacity += 1
                apple_x = None
                apple_y = None

    for i in range(1, len(l)):
        if l[i - 1] is None or l[i] is None:
            continue
        r, g, b = np.random.randint(0, 255, 3)
        cv2.line(frame, l[i], l[i - 1], (int(r), int(g), int(b)), thickness=int(len(l) / max_lc + 2) + 2)

    cv2.putText(frame, 'Score: ' + str(score), (450, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 203), 2)

    if flag == 1:
        cv2.putText(frame, 'YOU WIN !!', (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 3)

    cv2.imshow('Live Feed', frame)

    prev_c = center

    if cv2.waitKey(1) == 27:  # Press 'Esc' to exit
        break

# Release resources
cv2.destroyAllWindows()
cap.release()
