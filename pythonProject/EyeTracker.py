import cv2
import dlib
import numpy as np
from threading import Thread
import time


class EyeTracker:
    def __init__(self, scale_factor=1.0, min_neighbors=5):
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.cap = cv2.VideoCapture(0)
        self.is_running = False
        self.tracking = None

        self.lStart, self.lEnd = 42, 48
        self.rStart, self.rEnd = 36, 42
        self.sensitivity = 1.0

    def start_tracking(self):
        self.is_running = True
        tracking_thread = Thread(target=self._run_tracking_loop)
        tracking_thread.start()

    def stop_tracking(self):
        self.is_running = False
        self.cap.release()
        cv2.destroyAllWindows()

    def _run_tracking_loop(self):
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray, 0)

            for face in faces:
                shape = self.predictor(gray, face)
                shape = self.shape_to_np(shape)

                leftEye = shape[self.lStart:self.lEnd]
                rightEye = shape[self.rStart:self.rEnd]

                self._draw_eye_contours(frame, leftEye, rightEye)
                self._track_eye_movement(frame, leftEye, rightEye)

            cv2.imshow("Eye Tracking", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.stop_tracking()

    def shape_to_np(self, shape, dtype="int"):
        coords = np.zeros((68, 2), dtype=dtype)
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords

    def _draw_eye_contours(self, frame, leftEye, rightEye):
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

    def _track_eye_movement(self, frame, leftEye, rightEye):
        leftEyeCenter = np.mean(leftEye, axis=0)
        rightEyeCenter = np.mean(rightEye, axis=0)

        cv2.circle(frame, (int(leftEyeCenter[0]), int(leftEyeCenter[1])), 2, (0, 0, 255), -1)
        cv2.circle(frame, (int(rightEyeCenter[0]), int(rightEyeCenter[1])), 2, (0, 0, 255), -1)

        if self.tracking:
            self._update_tracking_lines(frame, leftEyeCenter, rightEyeCenter)

    def _update_tracking_lines(self, frame, leftEyeCenter, rightEyeCenter):
        cv2.line(frame, (int(self.tracking[0]), int(self.tracking[1])), (int(leftEyeCenter[0]), int(leftEyeCenter[1])),
                 (255, 0, 0), 2)
        cv2.line(frame, (int(self.tracking[2]), int(self.tracking[3])),
                 (int(rightEyeCenter[0]), int(rightEyeCenter[1])), (255, 0, 0), 2)

    def save_frame(self, frame):
        filename = f'eye_tracking_{int(time.time())}.jpg'
        cv2.imwrite(filename, frame)

    def process_video_file(self, file_path):
        video = cv2.VideoCapture(file_path)
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray, 0)

            for face in faces:
                shape = self.predictor(gray, face)
                shape = self.shape_to_np(shape)

                leftEye = shape[self.lStart:self.lEnd]
                rightEye = shape[self.rStart:self.rEnd]

                self._draw_eye_contours(frame, leftEye, rightEye)
                self._track_eye_movement(frame, leftEye, rightEye)

            cv2.imshow("Eye Tracking - Video", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video.release()
        cv2.destroyAllWindows()

        def toggle_fullscreen(self, window_name='Eye Tracking'):
            cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        def snapshot(self, window_name='Eye Tracking'):
            ret, frame = self.cap.read()
            if ret:
                cv2.imshow(window_name, frame)
                cv2.imwrite(f'snapshot_{int(time.time())}.jpg', frame)

        def apply_filter(self, filter_name="blur"):
            if filter_name == "blur":
                kernel_size = (15, 15)
                ret, frame = self.cap.read()
                if ret:
                    blurred_frame = cv2.GaussianBlur(frame, kernel_size, 0)
                    cv2.imshow("Filtered Frame", blurred_frame)

        def start_tracking_timeline(self):
            tracking_timeline = []
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.detector(gray, 0)

                for face in faces:
                    shape = self.predictor(gray, face)
                    shape = self.shape_to_np(shape)

                    leftEye = shape[self.lStart:self.lEnd]
                    rightEye = shape[self.rStart:self.rEnd]

                    self._draw_eye_contours(frame, leftEye, rightEye)
                    leftEyeCenter = np.mean(leftEye, axis=0)
                    rightEyeCenter = np.mean(rightEye, axis=0)
                    tracking_timeline.append((leftEyeCenter, rightEyeCenter))

                cv2.imshow('Eye Tracking - Timeline', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            self.is_running = False
            self._analyze_timeline(tracking_timeline)

        def _analyze_timeline(self, timeline):
            if len(timeline) < 2:
                return

            movements = [np.linalg.norm(np.array(timeline[i][0]) - np.array(timeline[i - 1][0])) +
                         np.linalg.norm(np.array(timeline[i][1]) - np.array(timeline[i - 1][1]))
                         for i in range(1, len(timeline))]

            average_movement = np.mean(movements)
            print(f"Movimento medio degli occhi: {average_movement:.2f} pixel")

        def set_tracking_sensitivity(self, sensitivity):
            self.sensitivity = sensitivity

        def get_eye_aspect_ratio(self, eye_points):
            A = np.linalg.norm(eye_points[1] - eye_points[5])
            B = np.linalg.norm(eye_points[2] - eye_points[4])
            C = np.linalg.norm(eye_points[0] - eye_points[3])
            return (A + B) / (2.0 * C)

        def reset_tracking(self):
            self.cap.release()
            self.cap = cv2.VideoCapture(0)
            self.is_running = False
            self.start_tracking()

        def not_implemented_yet(self):
            print("Questa funzione non è ancora stata implementata.")