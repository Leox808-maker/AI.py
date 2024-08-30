import tkinter as tk
from tkinter import ttk
from threading import Thread
import cv2
import numpy as np

class FaceRecognitionGUI:
    def __init__(self, recognizer):
        self.root = tk.Tk()
        self.root.title("Face Recognition GUI")
        self.recognizer = recognizer
        self.mode = tk.StringVar(value="Live Face Recognition")
        self.status_var = tk.StringVar(value="Status: Waiting")
        self.create_widgets()

    def create_widgets(self):
        self.mode_frame = ttk.LabelFrame(self.root, text="Select Mode")
        self.mode_frame.pack(fill="x", padx=10, pady=10)

        modes = [
            "Live Face Recognition",
            "Multiple Face Detection",
            "Face Capture",
            "Apply Filter",
            "Adjust Brightness",
            "Blur Faces",
            "Detect Eyes",
            "Detect Smile",
            "Record Video",
            "Detect Profile Face",
            "Recognize Face in Photo",
            "Face Detection in Video"
        ]

        for mode in modes:
            ttk.Radiobutton(self.mode_frame, text=mode, variable=self.mode, value=mode).pack(anchor=tk.W, padx=5, pady=2)

        self.action_frame = ttk.LabelFrame(self.root, text="Actions")
        self.action_frame.pack(fill="x", padx=10, pady=10)

        self.start_button = ttk.Button(self.action_frame, text="Start", command=self.start_recognition)
        self.start_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.stop_button = ttk.Button(self.action_frame, text="Stop", command=self.stop_recognition)
        self.stop_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.status_label = ttk.Label(self.root, textvariable=self.status_var)
        self.status_label.pack(fill="x", padx=10, pady=10)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def start_recognition(self):
        mode = self.mode.get()
        self.status_var.set(f"Status: Running - {mode}")
        if mode == "Live Face Recognition":
            self.run_thread(self.recognizer.start)
        elif mode == "Multiple Face Detection":
            self.run_thread(self.recognizer.detect_multiple_faces)
        elif mode == "Face Capture":
            self.run_thread(self.recognizer.capture_face)
        elif mode == "Apply Filter":
            self.run_thread(self.apply_filter)
        elif mode == "Adjust Brightness":
            self.run_thread(self.adjust_brightness)
        elif mode == "Blur Faces":
            self.run_thread(self.recognizer.blur_faces)
        elif mode == "Detect Eyes":
            self.run_thread(self.recognizer.detect_eyes)
        elif mode == "Detect Smile":
            self.run_thread(self.recognizer.detect_smile)
        elif mode == "Record Video":
            self.run_thread(self.recognizer.record_video)
        elif mode == "Detect Profile Face":
            self.run_thread(self.recognizer.detect_profile_face)
        elif mode == "Recognize Face in Photo":
            self.run_thread(self.recognizer.recognize_faces_in_photo)
        elif mode == "Face Detection in Video":
            self.run_thread(self.recognizer.detect_faces_in_video)

    def stop_recognition(self):
        self.recognizer.stop()
        self.status_var.set("Status: Stopped")

    def run_thread(self, target, *args):
        thread = Thread(target=target, args=args)
        thread.start()

        def apply_filter(self):
            filters = ["sepia", "negative", "gray"]
            selected_filter = tk.simpledialog.askstring("Filter", f"Choose filter ({', '.join(filters)}):")
            if selected_filter in filters:
                self.recognizer.apply_filter(selected_filter)
                self.status_var.set(f"Status: Applied {selected_filter} filter")
            else:
                self.status_var.set("Status: Invalid filter")

        def adjust_brightness(self):
            value = tk.simpledialog.askinteger("Brightness", "Enter brightness value (-100 to 100):")
            if isinstance(value, int) and -100 <= value <= 100:
                self.recognizer.adjust_brightness(value)
                self.status_var.set(f"Status: Brightness adjusted by {value}")
            else:
                self.status_var.set("Status: Invalid brightness value")

        def on_closing(self):
            self.stop_recognition()
            self.root.destroy()

        def run(self):
            self.root.mainloop()

    class FaceRecognizer:
        def __init__(self):
            self.cap = cv2.VideoCapture(0)
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.running = False
            self.frame = None
            self.faces = []

        def start(self):
            self.running = True
            while self.running:
                ret, self.frame = self.cap.read()
                if not ret:
                    continue
                gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
                self.faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                self._draw_faces()
                cv2.imshow('Face Recognizer', self.frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop()
                    break

        def stop(self):
            self.running = False
            self.cap.release()
            cv2.destroyAllWindows()

        def _draw_faces(self):
            for (x, y, w, h) in self.faces:
                cv2.rectangle(self.frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        def detect_multiple_faces(self):
            self.faces = []
            while self.running:
                ret, self.frame = self.cap.read()
                if not ret:
                    continue
                gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
                detected_faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                                                    minSize=(30, 30))
                if len(detected_faces) > 1:
                    self.faces = detected_faces
                    self._draw_faces()
                cv2.imshow('Multi-Face Recognizer', self.frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop()
                    break

        def capture_face(self):
            if self.faces is not None and len(self.faces) > 0:
                for i, (x, y, w, h) in enumerate(self.faces):
                    face_img = self.frame[y:y + h, x:x + w]
                    filename = f'captured_face_{i}.jpg'
                    cv2.imwrite(filename, face_img)
                    print(f'Captured face saved as {filename}')

        def apply_filter(self, filter_type='sepia'):
            if self.frame is not None:
                if filter_type == 'sepia':
                    kernel = np.array([[0.272, 0.534, 0.131],
                                       [0.349, 0.686, 0.168],
                                       [0.393, 0.769, 0.189]])
                    self.frame = cv2.transform(self.frame, kernel)
                elif filter_type == 'negative':
                    self.frame = cv2.bitwise_not(self.frame)
                elif filter_type == 'gray':
                    self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
                    self.frame = cv2.cvtColor(self.frame, cv2.COLOR_GRAY2BGR)
