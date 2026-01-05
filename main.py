import cv2
import easyocr
import time
import numpy as np

# Import modules from the src folder
from src.config import *
from src.math_utils import safe_evaluate
from src.hand_setup import get_hand_model, mp_hands, mp_drawing, mp_drawing_styles


def main():
    # Initialize EasyOCR reader
    print("Initializing OCR... Please wait.")
    reader = easyocr.Reader(['en'], gpu=True)

    # Setup webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    # Use actual dimensions from camera
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a blank canvas
    canvas = np.zeros((height, width, 3), np.uint8)

    # Timing and State variables
    prev_frame_time = 0
    xp, yp = 0, 0
    last_ocr_time = time.time()
    last_result = ""

    # Start MediaPipe hands detection
    with get_hand_model() as hands:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Flip and Convert
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    # Draw landmarks
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

                    # Extract points
                    points = [[int(lm.x * width), int(lm.y * height)] for lm in hand_landmarks.landmark]

                    if points:
                        # Extract key fingertips
                        x2, y2 = points[8]  # Index
                        x3, y3 = points[12]  # Middle

                        # Hand Orientation & Finger States
                        hand_label = handedness.classification[0].label

                        # Thumb Logic
                        if hand_label == 'Right':
                            fingers = [1 if points[TIP_IDS[0]][0] < points[TIP_IDS[0] - 1][0] else 0]
                        else:
                            fingers = [1 if points[TIP_IDS[0]][0] > points[TIP_IDS[0] - 1][0] else 0]

                        # Other Fingers Logic
                        fingers.extend(
                            [1 if points[TIP_IDS[id]][1] < points[TIP_IDS[id] - 2][1] else 0 for id in range(1, 5)])

                        # --- MODE SELECTION ---

                        # 1. PAUSE MODE (Index & Middle Extended)
                        if fingers[1] and fingers[2] and all(f == 0 for f in [fingers[0], fingers[3], fingers[4]]):
                            xp, yp = x2, y2
                            cv2.rectangle(frame, (x2 - 10, y2 - 15), (x3 + 10, y3 + 23), DRAW_COLOR, cv2.FILLED)
                            cv2.putText(frame, 'PAUSED',
                                        (width - cv2.getTextSize('PAUSED', FONT, 1, 2)[0][0] - 10, 30),
                                        FONT, 1, HUD_COLOR, 2, cv2.LINE_AA)

                        # 2. ERASE MODE (All fingers or 4 fingers extended)
                        if all(fingers) or all(fingers[1:]):
                            cv2.circle(frame, (x3, y3), 40, NO_COLOR, cv2.FILLED)
                            cv2.putText(frame, 'ERASE',
                                        (width - cv2.getTextSize('ERASE', FONT, 1, 2)[0][0] - 10, 30),
                                        FONT, 1, HUD_COLOR, 2, cv2.LINE_AA)
                            if xp == 0 and yp == 0:
                                xp, yp = x3, y3
                            cv2.circle(canvas, (x3, y3), 40, NO_COLOR, cv2.FILLED)
                            xp, yp = x3, y3

                        # 3. DRAW MODE (Only Index Extended)
                        if fingers[1] and all(f == 0 for f in [fingers[0], fingers[2], fingers[3], fingers[4]]):
                            cv2.circle(frame, (x2, y2), 10, DRAW_COLOR, cv2.FILLED)
                            cv2.putText(frame, 'DRAW',
                                        (width - cv2.getTextSize('DRAW', FONT, 1, 2)[0][0] - 10, 30),
                                        FONT, 1, HUD_COLOR, 2, cv2.LINE_AA)
                            if xp == 0 and yp == 0:
                                xp, yp = x2, y2
                            cv2.line(canvas, (xp, yp), (x2, y2), DRAW_COLOR, THICKNESS)
                            xp, yp = x2, y2
                        else:
                            xp, yp = 0, 0

            # Overlay Canvas
            frame[np.where(canvas != 0)] = canvas[np.where(canvas != 0)]

            # FPS Calculation
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time > 0 else 0
            prev_frame_time = new_frame_time
            cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), FONT, 1, HUD_COLOR, 2, cv2.LINE_AA)

            # --- OCR LOGIC ---
            if time.time() - last_ocr_time > 1:
                gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray_canvas, 50, 255, cv2.THRESH_BINARY_INV)
                detected_text = reader.readtext(thresh)

                text_content = "".join([t[1] for t in detected_text]).replace(" ", "")
                print("---\nDetected Text:", text_content)

                result = safe_evaluate(text_content)
                if result is not None:
                    last_result = f"Result: {result}"
                else:
                    last_result = "Result:"  # Reset or keep last valid? Keeping logic simple.

                last_ocr_time = time.time()

            # Display Result
            text_size, baseline = cv2.getTextSize(last_result, FONT, 1, 2)
            text_w, text_h = text_size
            text_x = (width - text_w) // 2
            text_y = height - 30

            cv2.rectangle(frame, (text_x - 10, text_y - text_h - baseline - 5),
                          (text_x + text_w + 20, text_y + baseline + 5), (0, 0, 0), -1)
            cv2.putText(frame, last_result, (text_x, text_y), FONT, 1, HUD_COLOR, 2, cv2.LINE_AA)

            cv2.imshow('Draw-and-Solve', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()