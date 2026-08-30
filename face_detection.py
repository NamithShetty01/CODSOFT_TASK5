import tkinter as tk
from tkinter import messagebox
import cv2
import os
import numpy as np
from PIL import Image, ImageTk


# =========================================================
# CODSOFT - FACE DETECTION AND RECOGNITION
# =========================================================

KNOWN_IMAGE_PATH = "known_faces/namith.jpg"
CAMERA_INDEX = 0

FACE_SIZE = (200, 200)
RECOGNITION_THRESHOLD = 120

VIDEO_WIDTH = 600
VIDEO_HEIGHT = 340


class FaceRecognitionApp:

    def __init__(self, root):

        self.root = root

        self.root.title(
            "CODSOFT - Face Detection and Recognition"
        )

        # IMPORTANT:
        # Smaller window so buttons are always visible
        self.root.geometry("850x650")
        self.root.resizable(False, False)
        self.root.configure(bg="#f4f4f4")

        # Camera
        self.camera = None
        self.camera_running = False
        self.photo = None

        # Model
        self.face_cascade = None
        self.recognizer = None
        self.model_loaded = False

        self.load_model()
        self.create_ui()

        # Keyboard controls
        self.root.bind("<Escape>", self.stop_camera)
        self.root.bind("<KeyPress-x>", self.exit_program)
        self.root.bind("<KeyPress-X>", self.exit_program)

        # Window X button
        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.exit_program
        )

    # =====================================================
    # LOAD FACE RECOGNITION MODEL
    # =====================================================

    def load_model(self):

        try:

            # Haar Cascade
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades +
                "haarcascade_frontalface_default.xml"
            )

            if self.face_cascade.empty():
                raise Exception(
                    "Could not load Haar Cascade."
                )

            # Check OpenCV contrib
            if not hasattr(cv2, "face"):
                raise Exception(
                    "OpenCV Face module is missing.\n\n"
                    "Run these commands:\n\n"
                    "pip uninstall opencv-python -y\n"
                    "pip install opencv-contrib-python"
                )

            # Create LBPH recognizer
            self.recognizer = (
                cv2.face.LBPHFaceRecognizer_create()
            )

            # Check known image
            if not os.path.exists(KNOWN_IMAGE_PATH):
                raise Exception(
                    "namith.jpg not found.\n\n"
                    "Put the image here:\n"
                    "known_faces/namith.jpg"
                )

            # Read image
            known_image = cv2.imread(
                KNOWN_IMAGE_PATH
            )

            if known_image is None:
                raise Exception(
                    "Could not read namith.jpg."
                )

            # Grayscale
            known_gray = cv2.cvtColor(
                known_image,
                cv2.COLOR_BGR2GRAY
            )

            # Detect face in known image
            known_faces = (
                self.face_cascade.detectMultiScale(
                    known_gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(60, 60)
                )
            )

            if len(known_faces) == 0:
                raise Exception(
                    "No face found in namith.jpg."
                )

            # First detected face
            x, y, w, h = known_faces[0]

            known_face = known_gray[
                y:y + h,
                x:x + w
            ]

            # Resize
            known_face = cv2.resize(
                known_face,
                FACE_SIZE
            )

            # Train
            self.recognizer.train(
                [known_face],
                np.array(
                    [1],
                    dtype=np.int32
                )
            )

            self.model_loaded = True

        except Exception as error:

            self.model_loaded = False

            messagebox.showerror(
                "Model Error",
                str(error)
            )

    # =====================================================
    # CREATE UI
    # =====================================================

    def create_ui(self):

        # -------------------------------------------------
        # HEADER
        # -------------------------------------------------

        header = tk.Frame(
            self.root,
            bg="#1f2937",
            height=105
        )

        header.pack(
            fill="x"
        )

        header.pack_propagate(False)

        title = tk.Label(
            header,
            text="👤  FACE DETECTION & RECOGNITION",
            font=("Arial", 23, "bold"),
            fg="white",
            bg="#1f2937"
        )

        title.pack(
            pady=(15, 2)
        )

        subtitle = tk.Label(
            header,
            text="Powered by OpenCV & LBPH Face Recognition",
            font=("Arial", 12),
            fg="white",
            bg="#1f2937"
        )

        subtitle.pack()

        # -------------------------------------------------
        # STATUS
        # -------------------------------------------------

        self.status_label = tk.Label(
            self.root,
            text=(
                "✓  Model loaded successfully"
                if self.model_loaded
                else "✗  Model could not be loaded"
            ),
            font=("Arial", 14, "bold"),
            fg="#222222",
            bg="#f4f4f4"
        )

        self.status_label.pack(
            pady=8
        )

        # -------------------------------------------------
        # CAMERA FRAME
        # -------------------------------------------------

        camera_frame = tk.Frame(
            self.root,
            bg="black",
            width=VIDEO_WIDTH,
            height=VIDEO_HEIGHT
        )

        camera_frame.pack(
            padx=20,
            pady=2
        )

        camera_frame.pack_propagate(False)

        self.camera_label = tk.Label(
            camera_frame,
            text="Camera is not running",
            font=("Arial", 20),
            fg="white",
            bg="black"
        )

        self.camera_label.pack(
            fill="both",
            expand=True
        )

        # -------------------------------------------------
        # FACE COUNT
        # -------------------------------------------------

        self.face_count_label = tk.Label(
            self.root,
            text="Faces Detected: 0",
            font=("Arial", 14, "bold"),
            fg="#222222",
            bg="#f4f4f4"
        )

        self.face_count_label.pack(
            pady=5
        )

        # -------------------------------------------------
        # BUTTONS
        # -------------------------------------------------

        button_frame = tk.Frame(
            self.root,
            bg="#f4f4f4"
        )

        button_frame.pack(
            pady=3
        )

        # START
        self.start_button = tk.Button(
            button_frame,
            text="📷  START CAMERA",
            command=self.start_camera,
            font=("Arial", 11, "bold"),
            bg="#1f2937",
            fg="white",
            activebackground="#374151",
            activeforeground="white",
            width=17,
            height=2,
            cursor="hand2"
        )

        self.start_button.grid(
            row=0,
            column=0,
            padx=6
        )

        # STOP
        self.stop_button = tk.Button(
            button_frame,
            text="⛔  STOP CAMERA",
            command=self.stop_camera,
            font=("Arial", 11, "bold"),
            bg="#555555",
            fg="white",
            activebackground="#444444",
            activeforeground="white",
            width=17,
            height=2,
            cursor="hand2"
        )

        self.stop_button.grid(
            row=0,
            column=1,
            padx=6
        )

        # EXIT
        self.exit_button = tk.Button(
            button_frame,
            text="❌  EXIT",
            command=self.exit_program,
            font=("Arial", 11, "bold"),
            bg="white",
            fg="black",
            activebackground="#dddddd",
            width=12,
            height=2,
            cursor="hand2"
        )

        self.exit_button.grid(
            row=0,
            column=2,
            padx=6
        )

        # -------------------------------------------------
        # INSTRUCTIONS
        # -------------------------------------------------

        instructions = tk.Label(
            self.root,
            text="ESC → Stop Camera     |     X → Exit",
            font=("Arial", 10),
            fg="#666666",
            bg="#f4f4f4"
        )

        instructions.pack(
            pady=5
        )

    # =====================================================
    # START CAMERA
    # =====================================================

    def start_camera(self):

        if not self.model_loaded:

            messagebox.showerror(
                "Error",
                "Face recognition model is not loaded."
            )

            return

        if self.camera_running:
            return

        # Windows camera
        self.camera = cv2.VideoCapture(
            CAMERA_INDEX,
            cv2.CAP_DSHOW
        )

        # Fallback
        if not self.camera.isOpened():

            self.camera.release()

            self.camera = cv2.VideoCapture(
                CAMERA_INDEX
            )

        if not self.camera.isOpened():

            self.camera = None

            self.status_label.config(
                text="✗  Could not open camera",
                fg="#b91c1c"
            )

            messagebox.showerror(
                "Camera Error",
                "Could not open the webcam.\n\n"
                "Check Windows camera permissions."
            )

            return

        # Camera resolution
        self.camera.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            640
        )

        self.camera.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            480
        )

        self.camera_running = True

        self.status_label.config(
            text="●  Camera running - Detecting faces...",
            fg="#008000"
        )

        self.start_button.config(
            state="disabled"
        )

        self.camera_label.config(
            text=""
        )

        self.update_camera()

    # =====================================================
    # CAMERA UPDATE
    # =====================================================

    def update_camera(self):

        if not self.camera_running:
            return

        if self.camera is None:
            return

        success, frame = self.camera.read()

        if not success:

            self.stop_camera()

            self.status_label.config(
                text="✗  Could not read camera frame",
                fg="#b91c1c"
            )

            return

        # Mirror camera
        frame = cv2.flip(
            frame,
            1
        )

        # Small preview
        frame = cv2.resize(
            frame,
            (VIDEO_WIDTH, VIDEO_HEIGHT)
        )

        # Grayscale
        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        # Detect faces
        faces = (
            self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=7,
                minSize=(60, 60)
            )
        )

        # -------------------------------------------------
        # PROCESS FACES
        # -------------------------------------------------

        for (x, y, w, h) in faces:

            face = gray[
                y:y + h,
                x:x + w
            ]

            try:

                face_resized = cv2.resize(
                    face,
                    FACE_SIZE
                )

                label, confidence = (
                    self.recognizer.predict(
                        face_resized
                    )
                )

                if (
                    label == 1
                    and
                    confidence <
                    RECOGNITION_THRESHOLD
                ):

                    name = "Namith"

                    box_color = (
                        0,
                        255,
                        0
                    )

                else:

                    name = "Unknown"

                    box_color = (
                        0,
                        0,
                        255
                    )

            except Exception:

                name = "Unknown"

                box_color = (
                    0,
                    0,
                    255
                )

            # Face rectangle
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                box_color,
                2
            )

            # Label
            cv2.putText(
                frame,
                name,
                (x, max(25, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                box_color,
                2,
                cv2.LINE_AA
            )

        # -------------------------------------------------
        # FACE COUNT
        # -------------------------------------------------

        cv2.putText(
            frame,
            f"Faces: {len(faces)}",
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        self.face_count_label.config(
            text=f"Faces Detected: {len(faces)}"
        )

        # -------------------------------------------------
        # DISPLAY IN TKINTER
        # -------------------------------------------------

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image = Image.fromarray(
            frame_rgb
        )

        self.photo = ImageTk.PhotoImage(
            image=image
        )

        self.camera_label.config(
            image=self.photo,
            text=""
        )

        # Continue
        if self.camera_running:

            self.root.after(
                20,
                self.update_camera
            )

    # =====================================================
    # STOP CAMERA
    # =====================================================

    def stop_camera(self, event=None):

        if not self.camera_running:
            return

        self.camera_running = False

        if self.camera is not None:

            self.camera.release()
            self.camera = None

        self.photo = None

        self.camera_label.config(
            image="",
            text="Camera stopped",
            font=("Arial", 20),
            fg="white",
            bg="black"
        )

        self.face_count_label.config(
            text="Faces Detected: 0"
        )

        self.status_label.config(
            text="✓  Camera stopped successfully",
            fg="#222222"
        )

        self.start_button.config(
            state="normal"
        )

    # =====================================================
    # EXIT
    # =====================================================

    def exit_program(self, event=None):

        self.camera_running = False

        if self.camera is not None:

            self.camera.release()
            self.camera = None

        self.root.destroy()


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    root = tk.Tk()

    app = FaceRecognitionApp(
        root
    )

    root.mainloop()