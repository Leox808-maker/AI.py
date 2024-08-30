import cv2
import numpy as np
import threading
import time

class FaceRecognizer:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.running = False
        self.frame = None
        self.faces = []

    def start(self):
        self.running = True
        thread = threading.Thread(target=self._update_frame)
        thread.start()

    def stop(self):
        self.running = False
        self.cap.release()
        cv2.destroyAllWindows()

    def _update_frame(self):
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

    def _draw_faces(self):
        for (x, y, w, h) in self.faces:
            cv2.rectangle(self.frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    def capture_face(self):
        if self.faces is not None and len(self.faces) > 0:
            for i, (x, y, w, h) in enumerate(self.faces):
                face_img = self.frame[y:y+h, x:x+w]
                filename = f'captured_face_{i}.jpg'
                cv2.imwrite(filename, face_img)
                print(f'Captured face saved as {filename}')

    def detect_multiple_faces(self):
        self.faces = []
        while self.running:
            ret, self.frame = self.cap.read()
            if not ret:
                continue
            gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            detected_faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            if len(detected_faces) > 1:
                self.faces = detected_faces
                self._draw_faces()
                self.capture_face()
            cv2.imshow('Multi-Face Recognizer', self.frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop()
                break

    def adjust_brightness(self, value):
        if self.frame is not None:
            hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            v = cv2.add(v, value)
            v = np.clip(v, 0, 255)
            hsv = cv2.merge((h, s, v))
            self.frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def blur_faces(self):
        if self.faces is not None and len(self.faces) > 0:
            for (x, y, w, h) in self.faces:
                roi = self.frame[y:y+h, x:x+w]
                blurred_roi = cv2.GaussianBlur(roi, (99, 99), 30)
                self.frame[y:y+h, x:x+w] = blurred_roi

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

    def change_resolution(self, width, height):
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)