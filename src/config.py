import cv2

# Dimensions & Camera
FRAME_WIDTH = 1024
FRAME_HEIGHT = 768

# Drawing Settings
THICKNESS = 5
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Colors (BGR Format)
HUD_COLOR = (0, 255, 0)    # Green
DRAW_COLOR = (0, 0, 255)   # Red
NO_COLOR = (0, 0, 0)       # Black (Eraser/Background)

# Hand Landmarks
TIP_IDS = [4, 8, 12, 16, 20]