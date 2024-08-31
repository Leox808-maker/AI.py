import cv2
import dlib
from scipy.spatial import distance as dist
import numpy as np
from threading import Thread
import time
import tkinter as tk
from tkinter import messagebox


class BlinkDetector:
    def __init__(self, threshold=0.25, consecutive_frames=3):
        self.EYE_AR_THRESH = threshold
        self.EYE_AR_CONSEC_FRAMES = consecutive_frames
        self.COUNTER = 0
        self.TOTAL = 0

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        self.lStart, self.lEnd = 42, 48
        self.rStart, self.rEnd = 36, 42

        self.cap = cv2.VideoCapture(0)
        self.is_running = False

    def eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)

    def start_detection(self):
        self.is_running = True
        detection_thread = Thread(target=self._run_detection_loop)
        detection_thread.start()

    def stop_detection(self):
        self.is_running = False
        self.cap.release()
        cv2.destroyAllWindows()

    def _run_detection_loop(self):
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = self.detector(gray, 0)

            for rect in rects:
                shape = self.predictor(gray, rect)
                shape = self.shape_to_np(shape)

                leftEye = shape[self.lStart:self.lEnd]
                rightEye = shape[self.rStart:self.rEnd]
                leftEAR = self.eye_aspect_ratio(leftEye)
                rightEAR = self.eye_aspect_ratio(rightEye)

                ear = (leftEAR + rightEAR) / 2.0

                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

                if ear < self.EYE_AR_THRESH:
                    self.COUNTER += 1
                else:
                    if self.COUNTER >= self.EYE_AR_CONSEC_FRAMES:
                        self.TOTAL += 1
                    self.COUNTER = 0

                cv2.putText(frame, "Blinks: {}".format(self.TOTAL), (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Frame", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.stop_detection()

        def shape_to_np(self, shape, dtype="int"):
            coords = np.zeros((68, 2), dtype=dtype)

        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords