import os
import re
import time

import cv2
import easyocr
import mediapipe as mp
import numpy as np
from sympy import sympify


def safe_evaluate(expr):
    expr = expr.replace("x", "*").replace("X", "*")  # Normalize multiplication symbols

    if not re.match(r'^[\d\s+\-*/^().]+$', expr):
        print("Invalid expression\n")
        return None

    try:
        safe_result = float(sympify(expr))  # Get precise result
        # Truncate to 2 decimal places without rounding
        safe_result = int(safe_result) if safe_result.is_integer() else "{:.2f}".format(safe_result).rstrip('0').rstrip(
            '.')
        print(f"Result: {safe_result}\n")
        return str(safe_result)
    except:
        print("Error in evaluation\n")
        return None


reader = easyocr.Reader(['en'], gpu=True)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

canvas = np.zeros((height, width, 3), np.uint8)  # Canvas for drawing

prev_frame_time = 0
new_frame_time = 0
font = cv2.FONT_HERSHEY_SIMPLEX

hudColor = (0, 255, 0)  # Green color for HUD
drawColor = (0, 0, 255)  # Red color for drawing
noColor = (0, 0, 0)  # Black color for erasing
thickness = 5  # Thickness of the painting
tipIds = [4, 8, 12, 16, 20]  # Fingertips indexes
xp, yp = 0, 0  # Previous finger position for continuous drawing

last_ocr_time = time.time()
last_result = ""

with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        frame = cv2.flip(frame, 1)  # Flip frame for natural interaction
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                points = []
                for lm in hand_landmarks.landmark:
                    points.append([int(lm.x * width), int(lm.y * height)])

                if len(points) != 0:
                    x1, y1 = points[4]  # Thumb tip
                    x2, y2 = points[8]  # Index finger tip
                    x3, y3 = points[12]  # Middle finger tip
                    x4, y4 = points[16]  # Ring finger tip
                    x5, y5 = points[20]  # Little finger tip

                    hand_label = handedness.classification[0].label  # 'Left' or 'Right'
                    fingers = []
                    if hand_label == 'Right':  # Adjust logic for right hand
                        fingers.append(1 if points[tipIds[0]][0] < points[tipIds[0] - 1][0] else 0)
                    else:  # Adjust logic for left hand (reverse)
                        fingers.append(1 if points[tipIds[0]][0] > points[tipIds[0] - 1][0] else 0)

                    for id in range(1, 5):
                        fingers.append(1 if points[tipIds[id]][1] < points[tipIds[id] - 2][1] else 0)

                    # Pause Drawing
                    if (fingers[1] and fingers[2]) and all(f == 0 for f in [fingers[0], fingers[3], fingers[4]]):
                        xp, yp = x2, y2
                        cv2.rectangle(frame, (x2 - 10, y2 - 15), (x3 + 10, y3 + 23), drawColor, cv2.FILLED)
                        cv2.putText(frame, 'PAUSED',
                                    (frame.shape[1] - cv2.getTextSize('PAUSED', font, 1, 2)[0][0] - 10, 30), font, 1,
                                    hudColor, 2, cv2.LINE_AA)

                    # Erase
                    if all(fingers) or all(fingers[1:]):
                        cv2.circle(frame, (x3, y3), 40, noColor, cv2.FILLED)  # Draw marker
                        cv2.putText(frame, 'ERASE',
                                    (frame.shape[1] - cv2.getTextSize('ERASE', font, 1, 2)[0][0] - 10, 30), font, 1,
                                    hudColor, 2, cv2.LINE_AA)
                        if xp == 0 and yp == 0:
                            xp, yp = x3, y3

                        cv2.circle(canvas, (x3, y3), 40, noColor, cv2.FILLED)  # Erase marker
                        xp, yp = x3, y3

                    # Draw
                    if fingers[1] and all(f == 0 for f in [fingers[0], fingers[2], fingers[3], fingers[4]]):
                        cv2.circle(frame, (x2, y2), 10, drawColor, cv2.FILLED)
                        cv2.putText(frame, 'DRAW',
                                    (frame.shape[1] - cv2.getTextSize('DRAW', font, 1, 2)[0][0] - 10, 30), font, 1,
                                    hudColor, 2, cv2.LINE_AA)
                        if xp == 0 and yp == 0:
                            xp, yp = x2, y2

                        cv2.line(canvas, (xp, yp), (x2, y2), drawColor, thickness)
                        xp, yp = x2, y2

                    else:
                        xp, yp = 0, 0

        frame[np.where(canvas != 0)] = canvas[np.where(canvas != 0)]
        new_frame_time = time.time()

        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time

        cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), font, 1, hudColor, 2, cv2.LINE_AA)

        if time.time() - last_ocr_time > 1:
            gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray_canvas, 50, 255, cv2.THRESH_BINARY_INV)

            detected_text = reader.readtext(thresh)
            text = "".join([text[1] for text in detected_text])
            text = text.replace(" ", "")
            print("---\nDetected Text:", text)

            result = safe_evaluate(text)
            if result is not None:
                last_result = f"Result: {result}"
            else:
                last_result = "Result:"

            last_ocr_time = time.time()

        # Draw black background for the result text
        text_size, baseline = cv2.getTextSize(last_result, font, 1, 2)
        text_width, text_height = text_size  # Unpack width and height
        text_x = (width - text_width) // 2
        text_y = height - 30  # Baseline for text
        # Draw background rectangle with correct height
        cv2.rectangle(frame,
                      (text_x - 10, text_y - text_height - baseline - 5),  # Proper top boundary
                      (text_x + text_width + 20, text_y + baseline + 5),  # Proper bottom boundary
                      (0, 0, 0),
                      -1)
        # Draw the text on top of the rectangle
        cv2.putText(frame, last_result, (text_x, text_y), font, 1, hudColor, 2, cv2.LINE_AA)

        cv2.imshow('Draw-and-Solve', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
