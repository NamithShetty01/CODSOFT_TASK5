# CODSOFT Face Detection and Recognition

This project is a Python desktop application that detects faces from a webcam and checks whether they match a known reference face stored in the project.

It uses:
- Python
- Tkinter for the GUI
- OpenCV for camera access and face detection
- LBPH Face Recognizer for face matching
- Pillow for image conversion in the Tkinter UI

## Features

- Starts and stops the webcam from a desktop app
- Detects faces in real time
- Draws bounding boxes around detected faces
- Identifies known face as "Namith"
- Marks unknown faces as "Unknown"
- Displays total face count in the live video feed
- Provides keyboard shortcuts for quick control

## Project Structure

```text
CODSOFT_TASK5/
├── face_detection.py
├── requirements.txt
├── known_faces/
│   └── namith.jpg
├── README.md
└── .git/
```

## Requirements

- Python 3.x
- Webcam or camera-enabled device
- OpenCV with face recognition support

Install dependencies:

```bash
pip install -r requirements.txt
```

If OpenCV face modules are missing, install:

```bash
pip uninstall opencv-python -y
pip install opencv-contrib-python
```

## Setup

1. Open a terminal in the project folder.
2. Create and activate a virtual environment (optional but recommended):

```bash
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Place your reference face image in:

```text
known_faces/namith.jpg
```

The app expects a clear frontal face image for matching.

## Run the Application

```bash
python face_detection.py
```

## Controls

- Start Camera: click the button in the interface
- Stop Camera: click the button or press `Esc`
- Exit: click the Exit button or press `X`

## Notes

- The app uses the built-in webcam index `0`.
- If the camera is not detected, make sure your device permissions are enabled.
- The recognition threshold can be adjusted in the code if needed.

## Main Code File

- `face_detection.py` contains the complete application logic, UI, and face recognition pipeline.

## License

This project is created for educational/demo purposes.
