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
