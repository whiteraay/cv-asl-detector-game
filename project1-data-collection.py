import os
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

import cv2
import mediapipe as mp
import numpy as np
import math
import time
LABEL = "Z"               
DATASET_DIR = f"/Users/aknur/Desktop/CVision/project-asl/data"
IMG_SIZE = 300
OFFSET = 20
MAX_IMAGES = 500
SAVE_DIR = os.path.join(DATASET_DIR, LABEL)
os.makedirs(SAVE_DIR, exist_ok=True)
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands( static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
counter = len(os.listdir(SAVE_DIR))

print("Dataset folder:", SAVE_DIR)
print("Starting counter:", counter)
print("Press 's' to save image")
print("Press 'q' to quit")
while True:
    success, frame = cap.read()
    if not success:
        continue
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    imgWhite = None
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            h_img, w_img, _ = frame.shape
            x_min, y_min = w_img, h_img
            x_max, y_max = 0, 0
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w_img), int(lm.y * h_img)
                x_min = min(x_min, x)
                x_max = max(x_max, x)
                y_min = min(y_min, y)
                y_max = max(y_max, y)
            w = x_max - x_min
            h = y_max - y_min
            x1 = max(0, x_min - OFFSET)
            y1 = max(0, y_min - OFFSET)
            x2 = min(w_img, x_max + OFFSET)
            y2 = min(h_img, y_max + OFFSET)
            imgCrop = frame[y1:y2, x1:x2]
            if imgCrop.size != 0:
                imgWhite = np.ones((IMG_SIZE, IMG_SIZE, 3), np.uint8) * 255
                aspectRatio = h / w
                if aspectRatio > 1:
                    k = IMG_SIZE / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, IMG_SIZE))
                    wGap = math.ceil((IMG_SIZE - wCal) / 2)
                    imgWhite[:, wGap:wGap + wCal] = imgResize
                else:
                    k = IMG_SIZE / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (IMG_SIZE, hCal))
                    hGap = math.ceil((IMG_SIZE - hCal) / 2)
                    imgWhite[hGap:hGap + hCal, :] = imgResize
                cv2.imshow("Crop", imgCrop)
                cv2.imshow("Processed", imgWhite)
    cv2.putText(frame, f"Label: {LABEL}", (20, 40),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(frame, f"Images: {counter}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    cv2.imshow("Camera", frame)
    key = cv2.waitKey(1)
    if key == ord("s") and imgWhite is not None: 
        cv2.imwrite(f"{SAVE_DIR}/{LABEL}_{int(time.time()*1000)}.jpg", imgWhite)
        counter += 1
        print("Saved:", counter)
        if counter >= MAX_IMAGES:
            print("Dataset complete")
            break
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
