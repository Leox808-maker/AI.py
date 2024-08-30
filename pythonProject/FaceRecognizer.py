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

        def flip_frame(self, direction=1):
            if self.frame is not None:
                self.frame = cv2.flip(self.frame, direction)

        def save_frame(self, filename='frame.jpg'):
            if self.frame is not None:
                cv2.imwrite(filename, self.frame)
                print(f'Frame saved as {filename}')

        def detect_eyes(self):
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            if self.faces is not None and len(self.faces) > 0:
                for (x, y, w, h) in self.faces:
                    roi_gray = cv2.cvtColor(self.frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
                    eyes = eye_cascade.detectMultiScale(roi_gray)
                    for (ex, ey, ew, eh) in eyes:
                        cv2.rectangle(self.frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)

        def detect_smile(self):
            smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
            if self.faces is not None and len(self.faces) > 0:
                for (x, y, w, h) in self.faces:
                    roi_gray = cv2.cvtColor(self.frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
                    smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=22,
                                                            minSize=(25, 25))
                    for (sx, sy, sw, sh) in smiles:
                        cv2.rectangle(self.frame, (x + sx, y + sy), (x + sx + sw, y + sy + sh), (0, 0, 255), 2)

        def record_video(self, filename='output.avi', duration=10):
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(filename, fourcc, 20.0, (640, 480))
            start_time = time.time()
            while self.running and (time.time() - start_time) < duration:
                ret, frame = self.cap.read()
                if not ret:
                    break
                out.write(frame)
                cv2.imshow('Recording', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            out.release()
            print(f'Video saved as {filename}')

        def toggle_face_detection(self):
            face_detection = True
            while self.running:
                ret, self.frame = self.cap.read()
                if not ret:
                    continue
                if face_detection:
                    gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
                    self.faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                                                    minSize=(30, 30))
                    self._draw_faces()
                cv2.imshow('Face Detection Toggle', self.frame)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    self.stop()
                    break
                elif key & 0xFF == ord('d'):
                    face_detection = not face_detection

        def zoom(self, scale=1.2):
            if self.frame is not None:
                height, width = self.frame.shape[:2]
                new_width = int(width / scale)
                new_height = int(height / scale)
                x1 = (width - new_width) // 2
                y1 = (height - new_height) // 2
                x2 = x1 + new_width
                y2 = y1 + new_height
                self.frame = cv2.resize(self.frame[y1:y2, x1:x2], (width, height))

        def pan(self, direction='right', step=10):
            if self.frame is not None:
                height, width = self.frame.shape[:2]
                if direction == 'right':
                    self.frame = np.roll(self.frame, step, axis=1)
                elif direction == 'left':
                    self.frame = np.roll(self.frame, -step, axis=1)
                elif direction == 'up':
                    self.frame = np.roll(self.frame, -step, axis=0)
                elif direction == 'down':
                    self.frame = np.roll(self.frame, step, axis=0)

                    def detect_faces_in_video(self, video_path):
                        video = cv2.VideoCapture(video_path)
                        while video.isOpened():
                            ret, frame = video.read()
                            if not ret:
                                break
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                                                       minSize=(30, 30))
                            for (x, y, w, h) in faces:
                                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                            cv2.imshow('Face Detection in Video', frame)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break
                        video.release()
                        cv2.destroyAllWindows()

                    def take_photo(self, filename='photo.jpg'):
                        if self.frame is not None:
                            cv2.imwrite(filename, self.frame)
                            print(f'Photo saved as {filename}')

                    def detect_profile_face(self):
                        profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
                        while self.running:
                            ret, self.frame = self.cap.read()
                            if not ret:
                                continue
                            gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
                            profiles = profile_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                                                        minSize=(30, 30))
                            for (x, y, w, h) in profiles:
                                cv2.rectangle(self.frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                            cv2.imshow('Profile Face Detection', self.frame)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                self.stop()
                                break

                    def recognize_faces_in_photo(self, photo_path):
                        img = cv2.imread(photo_path)
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                                                   minSize=(30, 30))
                        for (x, y, w, h) in faces:
                            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        cv2.imshow('Face Recognition in Photo', img)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
