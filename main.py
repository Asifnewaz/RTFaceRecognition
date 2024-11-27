import math
import os
import sys
import cv2
import numpy as np
import face_recognition
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QFrame,
)
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtCore import Qt, QTimer


def face_confidence(face_distance, face_match_threshold=0.6):
    frange = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (frange * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return round(value, 2)  # Return float value for easier comparison


class MainWindow(QMainWindow):
    face_locations = []
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    process_current_frame = True

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Recognition App")
        self.setGeometry(100, 100, 800, 600)  # Set main window size
        self.initUI()
        self.encode_faces()

        # Initialize the camera
        self.cap = cv2.VideoCapture(0)

        # Timer for processing frames
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.run_recognition)
        self.timer.start(30)  # Update every 30ms

    def encode_faces(self):
        for image in os.listdir('faces'):
            face_image = face_recognition.load_image_file(f'faces/{image}')
            face_encoding = face_recognition.face_encodings(face_image)[0]

            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(image.split('.')[0])  # Remove file extension

        print(self.known_face_names)

    def initUI(self):
        main_widget = QWidget()
        self.main_layout = QHBoxLayout()

        # Left - Camera View
        self.camera_view = QLabel("Initializing Camera...")
        self.camera_view.setFrameStyle(QFrame.Box)
        self.camera_view.setAlignment(Qt.AlignCenter)
        self.camera_view.setStyleSheet("background-color: lightgray; color: black;")
        self.camera_view.setFixedSize(640, 480)  # Set initial frame size

        # Right - Info and Buttons
        right_layout = QVBoxLayout()

        self.name_label = QLabel("Name: ")
        self.id_label = QLabel("ID: ")
        self.time_label = QLabel("Time: ")

        for label in [self.name_label, self.id_label, self.time_label]:
            label.setStyleSheet("font-size: 14px;")

        right_layout.addWidget(self.name_label)
        right_layout.addWidget(self.id_label)
        right_layout.addWidget(self.time_label)
        right_layout.addStretch()

        button_layout = QHBoxLayout()
        save_button = QPushButton()
        save_button.setIcon(QIcon.fromTheme("document-save"))
        save_button.setFixedSize(40, 40)
        save_button.setStyleSheet("background-color: white; border: none;")

        delete_button = QPushButton()
        delete_button.setIcon(QIcon.fromTheme("edit-delete"))
        delete_button.setFixedSize(40, 40)
        delete_button.setStyleSheet("background-color: white; border: none;")

        button_layout.addWidget(save_button)
        button_layout.addWidget(delete_button)
        right_layout.addLayout(button_layout)

        self.main_layout.addWidget(self.camera_view, stretch=3)
        self.main_layout.addLayout(right_layout, stretch=1)

        main_widget.setLayout(self.main_layout)
        self.setCentralWidget(main_widget)

    def run_recognition(self):
        ret, frame = self.cap.read()
        if not ret:
            self.camera_view.setText("Failed to capture frame")
            return

        if self.process_current_frame:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            self.face_locations = face_recognition.face_locations(rgb_small_frame)
            self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

            self.face_names = []
            for face_encoding in self.face_encodings:
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)

                if len(face_distances) > 0:  # Check if face_distances is not empty
                    best_match_index = np.argmin(face_distances)
                    name = "Unknown"
                    confidence = 0  # Default confidence

                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confidence = face_confidence(face_distances[best_match_index])

                    self.face_names.append((name, confidence))  # Append tuple
                else:
                    self.face_names.append(("Unknown", 0))

        self.process_current_frame = not self.process_current_frame

        for (top, right, bottom, left), (name, confidence) in zip(self.face_locations, self.face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            if confidence > 80:  # Confidence threshold for green border
                color = (0, 255, 0)  # Green
                self.name_label.setText(f"Name: {name}")
                self.time_label.setText(f"Time: {datetime.now().strftime('%H:%M:%S')}")  # Update time
            else:
                color = (0, 0, 255)  # Red
                self.name_label.setText("Name: Unknown")
                self.time_label.setText("Time: -")  # Reset time if no valid face is detected

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, -1)
            cv2.putText(frame, f"{name} ({confidence}%)", (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8,
                        (255, 255, 255), 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb_frame.shape
        q_image = QImage(rgb_frame.data, width, height, channel * width, QImage.Format_RGB888)
        self.camera_view.setPixmap(QPixmap.fromImage(q_image))

    def closeEvent(self, event):
        self.cap.release()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()