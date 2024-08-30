import cv2
from deepface import DeepFace
import numpy as np
import time
import tkinter as tk
from tkinter import messagebox



class EmotionRecognizer:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.model = None
        self.emotions = ["happy", "sad", "neutral", "angry", "surprise", "fear", "disgust"]
        self.is_running = False
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def load_model(self):
        self.model = DeepFace.build_model("Emotion")

    def start_recognition(self):
        if self.model is None:
            self.load_model()
        self.is_running = True
        self._run_recognition_loop()

    def _run_recognition_loop(self):
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                print("Errore nella cattura del frame")
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                face = frame[y:y + h, x:x + w]
                emotion, confidence = self._predict_emotion(face)
                self._display_emotion(frame, x, y, w, h, emotion, confidence)

            cv2.imshow('Riconoscimento Emozioni', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.is_running = False

        self.stop_recognition()

    def _predict_emotion(self, face):
        try:
            result = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)
            emotion = result['dominant_emotion']
            confidence = result['emotion'][emotion]
        except Exception as e:
            print(f"Errore durante il riconoscimento dell'emozione: {e}")
            emotion = "unknown"
            confidence = 0.0

        return emotion, confidence

    def _display_emotion(self, frame, x, y, w, h, emotion, confidence):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        emotion_text = f'{emotion}: {confidence:.2f}%'
        cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    def stop_recognition(self):
        self.is_running = False
        self.cap.release()
        cv2.destroyAllWindows()

    def save_frame(self, frame, emotion):
        filename = f'emotion_{emotion}_{int(time.time())}.jpg'
        cv2.imwrite(filename, frame)
        print(f"Frame salvato come {filename}")

    def set_emotion_threshold(self, threshold):
        self.emotion_threshold = threshold

    def process_video_file(self, file_path):
        video = cv2.VideoCapture(file_path)
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                face = frame[y:y + h, x:x + w]
                emotion, confidence = self._predict_emotion(face)
                self._display_emotion(frame, x, y, w, h, emotion, confidence)

            cv2.imshow('Riconoscimento Emozioni - Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video.release()
        cv2.destroyAllWindows()

    def capture_emotion(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Errore nella cattura del frame")
            return

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face = frame[y:y + h, x:x + w]
            emotion, confidence = self._predict_emotion(face)
            self._display_emotion(frame, x, y, w, h, emotion, confidence)
            self.save_frame(frame, emotion)

        cv2.imshow('Cattura Emozione', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        def recognize_emotion_from_image(self, image_path):
            image = cv2.imread(image_path)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                face = image[y:y + h, x:x + w]
                emotion, confidence = self._predict_emotion(face)
                self._display_emotion(image, x, y, w, h, emotion, confidence)

            cv2.imshow('Riconoscimento Emozione - Immagine', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        def set_capture_resolution(self, width, height):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        def set_video_source(self, source=0):
            self.cap = cv2.VideoCapture(source)

        def reset_recognition(self):
            self.cap.release()
            self.cap = cv2.VideoCapture(0)
            self.is_running = False

        def toggle_fullscreen(self, window_name='Riconoscimento Emozioni'):
            cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        def snapshot(self, window_name='Riconoscimento Emozioni'):
            ret, frame = self.cap.read()
            if ret:
                cv2.imshow(window_name, frame)
                cv2.imwrite(f'snapshot_{int(time.time())}.jpg', frame)
                print(f"Snapshot salvato come snapshot_{int(time.time())}.jpg")

        def apply_filter(self, filter_name="blur"):
            if filter_name == "blur":
                kernel_size = (15, 15)
                ret, frame = self.cap.read()
                if ret:
                    blurred_frame = cv2.GaussianBlur(frame, kernel_size, 0)
                    cv2.imshow("Filtered Frame", blurred_frame)

        def start_emotion_timeline(self):
            emotion_timeline = []
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    print("Errore nella cattura del frame")
                    break

                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_detector.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5,
                                                            minSize=(30, 30))

                for (x, y, w, h) in faces:
                    face = frame[y:y + h, x:x + w]
                    emotion, confidence = self._predict_emotion(face)
                    emotion_timeline.append((emotion, confidence))
                    self._display_emotion(frame, x, y, w, h, emotion, confidence)

                cv2.imshow('Riconoscimento Emozioni - Timeline', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            self.is_running = False
            self._analyze_timeline(emotion_timeline)

        def _analyze_timeline(self, timeline):
            emotion_counts = {emotion: 0 for emotion in self.emotions}
            for emotion, _ in timeline:
                if emotion in emotion_counts:
                    emotion_counts[emotion] += 1

            print("Analisi delle emozioni rilevate durante la sessione:")
            for emotion, count in emotion_counts.items():
                print(f"{emotion}: {count} volte")

        def not_implemented_yet(self):
            messagebox.showinfo("Non Implementato", "Questa funzione non è ancora stata implementata.")
