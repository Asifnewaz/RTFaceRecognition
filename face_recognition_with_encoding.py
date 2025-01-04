import pickle

import face_recognition
from keras.src.models import Sequential
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.src.saving import load_model
from keras.src.utils import to_categorical
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QMessageBox, QStackedWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, Qt
import os
import time

class CountdownThread(QThread):
    countdown_signal = pyqtSignal(int)

    def run(self):
        for i in range(3, 0, -1):
            self.countdown_signal.emit(i)
            self.sleep(1)

class FaceRecognitionApp(QWidget):
    def __init__(self):
        super().__init__()

        # Initialize main stacked widget
        self.stack = QStackedWidget()
        self.setWindowTitle("Face Recognition App")
        self.setGeometry(100, 100, 800, 600)

        # Create pages
        self.start_page = self.create_start_page()
        self.video_feed_page = self.create_video_feed_page()
        self.countdown_label = QLabel("", self)

        # Add pages to stack
        self.stack.addWidget(self.start_page)
        self.stack.addWidget(self.video_feed_page)

        # Set layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.stack)
        self.setLayout(main_layout)

        # Load the encoding file
        print("Loading Encode File ...")
        file = open('EncodeFile.p', 'rb')
        encodeListKnownWithIds = pickle.load(file)
        file.close()
        self.encodeListKnown, self.studentIds = encodeListKnownWithIds


        self.detected_faces = set()
        self.pause_processing = False  # Initialize pause_processing
        self.processing = False  # Initialize processing
        self.cap = None
        self.timer = None

    def create_start_page(self):
        start_page = QWidget()
        layout = QVBoxLayout()

        # Class ID Input
        self.class_id_input = QLineEdit()
        self.class_id_input.setPlaceholderText("Enter Class ID")
        self.class_id_input.setStyleSheet("font-size: 16px; padding: 5px;")
        layout.addWidget(self.class_id_input)

        # Start Button
        start_button = QPushButton("Start")
        start_button.setStyleSheet("font-size: 16px; padding: 10px; background-color: green; color: white;")
        start_button.clicked.connect(self.start_countdown)
        layout.addWidget(start_button)

        start_page.setLayout(layout)
        return start_page

    def create_video_feed_page(self):
        video_feed_page = QWidget()
        layout = QHBoxLayout()

        # Video Feed Layout
        video_layout = QVBoxLayout()
        self.video_feed = QLabel()
        self.video_feed.setFixedSize(640, 480)
        video_layout.addWidget(self.video_feed)

        # Info Layout
        info_layout = QVBoxLayout()
        self.id_label = QLabel("ID: ")
        self.id_label.setStyleSheet("font-size: 16px;")
        info_layout.addWidget(self.id_label)

        self.time_label = QLabel("Time: ")
        self.time_label.setStyleSheet("font-size: 16px;")
        info_layout.addWidget(self.time_label)

        self.loader_label = QLabel("Processing...")
        self.loader_label.setStyleSheet("font-size: 16px; color: blue;")
        self.loader_label.setVisible(False)
        info_layout.addWidget(self.loader_label)

        layout.addLayout(video_layout)
        layout.addLayout(info_layout)
        video_feed_page.setLayout(layout)

        return video_feed_page

    def start_countdown(self):
        self.class_id = self.class_id_input.text().strip()
        if not self.class_id:
            QMessageBox.warning(self, "Error", "Please enter a valid Class ID.")
            return

        self.countdown_label.setText("Starting in...")
        self.countdown_label.setAlignment(Qt.AlignCenter)
        self.countdown_label.setStyleSheet("font-size: 24px; color: red;")
        self.stack.addWidget(self.countdown_label)
        self.stack.setCurrentWidget(self.countdown_label)

        self.countdown_thread = CountdownThread()
        self.countdown_thread.countdown_signal.connect(self.update_countdown)
        self.countdown_thread.finished.connect(self.start_video_feed)
        self.countdown_thread.start()

    def update_countdown(self, value):
        self.countdown_label.setText( "Loading data . . . " + str(value))

    def start_video_feed(self):
        self.stack.setCurrentWidget(self.video_feed_page)

        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_feed)
        self.timer.start(10)

    def update_feed(self):

        ret, frame = self.cap.read()

        if ret:
            # Resize frame to center the face
            height, width, _ = frame.shape
            crop_size = min(height, width)
            start_x = (width - crop_size) // 2
            start_y = (height - crop_size) // 2
            frame = frame[start_y:start_y + crop_size, start_x:start_x + crop_size]
            frame = cv2.resize(frame, (640, 480))

            # Update video feed
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.video_feed.setPixmap(QPixmap.fromImage(qimg))

            if self.pause_processing:
                return

            # If not processing, trigger detection
            if not self.processing:
                self.processing = True
                self.loader_label.setVisible(True)
                QTimer.singleShot(500, lambda: self.handle_detection(frame))

    def handle_detection(self, frame):
        self.pause_processing = True
        result = self.process_frame(frame)

        if result == "Unknown":
            # Clear labels if no face is detected
            self.id_label.setText("ID: ")
            self.time_label.setText("Time: ")
            self.loader_label.setVisible(False)
        else:

            self.id_label.setText(f"ID: {result}")
            self.time_label.setText(f"Time: {time.strftime('%H:%M:%S')}")
            self.loader_label.setVisible(False)

            if result in self.detected_faces:

                # Show message if attendance is already saved
                msg = QMessageBox()
                msg.setWindowTitle("Attendance Saved")
                msg.setText("User already marked present.")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
            else:
                # Save attendance and update labels
                self.detected_faces.add(result)
                # Show success message
                msg = QMessageBox()
                msg.setWindowTitle("Recognition Success")
                msg.setText("User recognized successfully!")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()

        # Resume processing
        self.processing = False
        self.pause_processing = False


    def process_frame(self, image):
        # Convert BGR (OpenCV) to RGB (face_recognition library)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # # Find all face locations in the frame
        face_locations = face_recognition.face_locations(rgb_image)
        encode_cur_frame = face_recognition.face_encodings(rgb_image, face_locations)

        result = "Unknown"

        if face_locations:
            for encodeFace, faceLoc in zip(encode_cur_frame, face_locations):
                matches = face_recognition.compare_faces(self.encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(self.encodeListKnown, encodeFace)
                # print("matches", matches)
                # print("faceDis", faceDis)

                match_index = np.argmin(faceDis)
                # print("Match Index", matchIndex)

                if matches[match_index]:
                    # print("Known Face Detected")
                    self.id = self.studentIds[match_index]
                    top, right, bottom, left = faceLoc
                    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

                    # Display text (e.g., 'Face Detected') at the top-left corner of the face
                    result =   self.id
                print(self.id)

        return result

    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication([])
    window = FaceRecognitionApp()
    window.show()
    app.exec_()