import re
import time

import cv2
import easyocr
import mediapipe as mp
import numpy as np
from sympy import sympify


# Function to safely evaluate a mathematical expression using SymPy
def safe_evaluate(expr):
    expr = expr.replace("x", "*").replace("X", "*")  # Normalize multiplication symbols
    # Validate expression to contain only numbers and allowed operators
    if not re.match(r'^[\d\s+\-*/^().]+$', expr):
        print("Invalid expression\n")
        return None
    try:
        safe_result = float(sympify(expr))  # Evaluate the expression using SymPy
        # Format result: integer if whole number, otherwise truncate decimals to 2 places
        safe_result = int(safe_result) if safe_result.is_integer() else "{:.2f}".format(safe_result).rstrip('0').rstrip(
            '.')
        print(f"Result: {safe_result}\n")
        return str(safe_result)
    except:
        print("Error in evaluation\n")
        return None


# Initialize EasyOCR reader for English with GPU enabled
reader = easyocr.Reader(['en'], gpu=True)

# Initialize MediaPipe drawing utilities and hand detection modules
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Setup webcam capture and set frame dimensions
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create a blank canvas for drawing (not opened in its own window)
canvas = np.zeros((height, width, 3), np.uint8)

# Initialize timing variables for FPS calculation and OCR updates
prev_frame_time = 0
new_frame_time = 0
font = cv2.FONT_HERSHEY_SIMPLEX

# Define colors and drawing parameters
hudColor = (0, 255, 0)  # Color for HUD elements
drawColor = (0, 0, 255)  # Color for drawing strokes
noColor = (0, 0, 0)  # Color used for erasing (black)
thickness = 5  # Thickness of drawn lines
tipIds = [4, 8, 12, 16, 20]  # Indexes of finger tip landmarks
xp, yp = 0, 0  # Previous finger coordinates for continuous drawing

last_ocr_time = time.time()  # Timer to control OCR update frequency
last_result = ""  # String to store the last evaluated result

# Start MediaPipe hands detection
with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the frame horizontally for natural interaction
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        results = hands.process(frame_rgb)  # Process the frame for hand landmarks

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Draw hand landmarks and connections on the frame
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                # Extract landmark points and scale them according to frame dimensions
                points = [[int(lm.x * width), int(lm.y * height)] for lm in hand_landmarks.landmark]

                if points:
                    # Extract coordinates of key fingertips
                    x1, y1 = points[4]  # Thumb tip
                    x2, y2 = points[8]  # Index finger tip
                    x3, y3 = points[12]  # Middle finger tip
                    x4, y4 = points[16]  # Ring finger tip
                    x5, y5 = points[20]  # Little finger tip

                    # Determine hand orientation (Right or Left) and detect if finger is extended
                    hand_label = handedness.classification[0].label
                    # For the thumb, adjust based on hand label
                    fingers = [1 if points[tipIds[0]][0] < points[tipIds[0] - 1][0] else 0] if hand_label == 'Right' \
                        else [1 if points[tipIds[0]][0] > points[tipIds[0] - 1][0] else 0]

                    # Check for extension of other fingers by comparing landmark positions
                    fingers.extend(
                        [1 if points[tipIds[id]][1] < points[tipIds[id] - 2][1] else 0 for id in range(1, 5)])

                    # Pause mode: if index and middle fingers are extended and others are not
                    if fingers[1] and fingers[2] and all(f == 0 for f in [fingers[0], fingers[3], fingers[4]]):
                        xp, yp = x2, y2
                        cv2.rectangle(frame, (x2 - 10, y2 - 15), (x3 + 10, y3 + 23), drawColor, cv2.FILLED)
                        cv2.putText(frame, 'PAUSED',
                                    (frame.shape[1] - cv2.getTextSize('PAUSED', font, 1, 2)[0][0] - 10, 30),
                                    font, 1, hudColor, 2, cv2.LINE_AA)

                    # Erase mode: if all fingers are extended or if all except thumb are extended
                    if all(fingers) or all(fingers[1:]):
                        cv2.circle(frame, (x3, y3), 40, noColor, cv2.FILLED)
                        cv2.putText(frame, 'ERASE',
                                    (frame.shape[1] - cv2.getTextSize('ERASE', font, 1, 2)[0][0] - 10, 30),
                                    font, 1, hudColor, 2, cv2.LINE_AA)
                        if xp == 0 and yp == 0:
                            xp, yp = x3, y3
                        # Erase on the canvas by drawing a filled circle in noColor
                        cv2.circle(canvas, (x3, y3), 40, noColor, cv2.FILLED)
                        xp, yp = x3, y3

                    # Draw mode: if only index finger is extended
                    if fingers[1] and all(f == 0 for f in [fingers[0], fingers[2], fingers[3], fingers[4]]):
                        cv2.circle(frame, (x2, y2), 10, drawColor, cv2.FILLED)
                        cv2.putText(frame, 'DRAW',
                                    (frame.shape[1] - cv2.getTextSize('DRAW', font, 1, 2)[0][0] - 10, 30),
                                    font, 1, hudColor, 2, cv2.LINE_AA)
                        if xp == 0 and yp == 0:
                            xp, yp = x2, y2
                        # Draw a line on the canvas from previous point to current index finger tip
                        cv2.line(canvas, (xp, yp), (x2, y2), drawColor, thickness)
                        xp, yp = x2, y2
                    else:
                        xp, yp = 0, 0

        # Overlay the canvas on the current frame
        frame[np.where(canvas != 0)] = canvas[np.where(canvas != 0)]
        # Calculate FPS for performance monitoring
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), font, 1, hudColor, 2, cv2.LINE_AA)

        # Perform OCR every second to extract drawn mathematical expression
        if time.time() - last_ocr_time > 1:
            gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)  # Convert canvas to grayscale
            _, thresh = cv2.threshold(gray_canvas, 50, 255,
                                      cv2.THRESH_BINARY_INV)  # Apply thresholding for clear contrast
            detected_text = reader.readtext(thresh)  # Read text using EasyOCR
            text = "".join([text[1] for text in detected_text]).replace(" ", "")
            print("---\nDetected Text:", text)
            result = safe_evaluate(text)  # Evaluate the extracted text
            last_result = f"Result: {result}" if result is not None else "Result:"
            last_ocr_time = time.time()

        # Display the evaluated result on the frame with a black background
        text_size, baseline = cv2.getTextSize(last_result, font, 1, 2)
        text_width, text_height = text_size
        text_x = (width - text_width) // 2
        text_y = height - 30
        cv2.rectangle(frame, (text_x - 10, text_y - text_height - baseline - 5),
                      (text_x + text_width + 20, text_y + baseline + 5), (0, 0, 0), -1)
        cv2.putText(frame, last_result, (text_x, text_y), font, 1, hudColor, 2, cv2.LINE_AA)

        # Show the final output frame with the canvas overlay and HUD elements
        cv2.imshow('Draw-and-Solve', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the webcam and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()
