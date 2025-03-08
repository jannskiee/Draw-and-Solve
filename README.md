# Draw-and-Solve: A Real-Time Virtual Handwritten Math Expression Recognition and Evaluation System

## Project Overview

This project enables users to draw mathematical expressions using hand gestures, which are then recognized and evaluated in real-time. It leverages:

- **MediaPipe** for hand gesture tracking
- **EasyOCR** for handwritten text recognition
- **SymPy** for safely evaluating mathematical expressions
- **OpenCV** for real-time video processing and drawing

## Features

- **Real-time hand tracking**: Detects gestures for drawing, pausing, and erasing.
- **Mathematical expression recognition**: Recognizes handwritten equations.
- **Safe expression evaluation**: Computes results using SymPy with error handling.
- **HUD overlay**: Displays FPS, current operation mode, and evaluated result.

## Requirements

### Dependencies

Ensure you have the following installed:

```sh
pip install opencv-python numpy easyocr mediapipe sympy
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Hardware

- Webcam for real-time capture
- GPU (optional, for EasyOCR acceleration)

## Running the Project

1. **Clone the repository**:

```sh
git clone <repository_url>
cd draw-and-solve
```

2. **Run the script**:

```sh
python draw_and_solve.py
```

3. **Interact using gestures**:
   - **Index finger extended** → Draw
   - **All fingers extended** → Erase
   - **Index & middle fingers extended** → Pause
4. **View results**: The recognized mathematical expression is evaluated and displayed.

## Exiting the Program

Press **'q'** to exit the application.

## Notes

- Ensure a well-lit environment for accurate hand tracking.
- For better OCR accuracy, write expressions clearly and distinctly.

