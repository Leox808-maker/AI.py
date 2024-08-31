import cv2
import numpy as np


class EyeTrackingCalibrator:
    def __init__(self, screen_width=1920, screen_height=1080):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.calibration_points = []
        self.window_name = "Eye Tracking Calibration"
        self.shapes = ["circle", "square", "triangle"]

    def start_calibration(self):
        self.calibration_points = []
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.screen_width, self.screen_height)

        for shape in self.shapes:
            self.display_shape(shape)
            cv2.waitKey(2000)  # Display each shape for 2 seconds

        cv2.destroyAllWindows()
        print("Calibration complete.")
        self.show_calibration_points()

    def display_shape(self, shape):
        frame = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
        center = (self.screen_width // 2, self.screen_height // 2)
        size = 100

        if shape == "circle":
            cv2.circle(frame, center, size, (0, 255, 0), -1)
        elif shape == "square":
            top_left = (center[0] - size, center[1] - size)
            bottom_right = (center[0] + size, center[1] + size)
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), -1)
        elif shape == "triangle":
            pts = np.array([[center[0], center[1] - size],
                            [center[0] - size, center[1] + size],
                            [center[0] + size, center[1] + size]], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(frame, [pts], (0, 255, 0))
        else:
            raise ValueError("Unknown shape type")

        cv2.imshow(self.window_name, frame)
        self.calibration_points.append(center)

    def show_calibration_points(self):
        print("Calibration points:")
        for idx, point in enumerate(self.calibration_points):
            print(f"Point {idx + 1}: {point}")


if __name__ == "__main__":
    calibrator = EyeTrackingCalibrator()
    calibrator.start_calibration()

    def save_calibration_points(self):
        with open(self.save_path, 'w') as file:
            for point in self.calibration_points:
                file.write(f"{point[0]},{point[1]}\n")
        print(f"Calibration points saved to {self.save_path}")

    def load_calibration_points(self):
        self.calibration_points = []
        try:
            with open(self.save_path, 'r') as file:
                for line in file:
                    x, y = map(int, line.strip().split(','))
                    self.calibration_points.append((x, y))
            print(f"Calibration points loaded from {self.save_path}")
        except FileNotFoundError:
            print(f"No calibration points file found at {self.save_path}")

    def repeat_calibration(self):
        print("Repeating calibration...")
        self.start_calibration()

    def clear_calibration_points(self):
        self.calibration_points = []
        print("Calibration points cleared.")

    def draw_text(self, frame, text, position, color=(255, 255, 255), font_scale=1, thickness=2):
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

    def show_calibration_points(self):
        frame = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
        for point in self.calibration_points:
            cv2.circle(frame, point, 10, (0, 0, 255), -1)
        self.draw_text(frame, "Calibration Points", (10, 30), color=(0, 255, 0))
        cv2.imshow("Calibration Points", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
