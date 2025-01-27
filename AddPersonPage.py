import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QMessageBox, QStackedWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, Qt
import os
from TrainData import TrainData

class AddPersonPage(QWidget):
    def __init__(self, class_id):
        self.class_id = class_id
        super().__init__()
        self.init_ui()
        self.image_count = 0
        self.capture_images = False
        self.cap = cv2.VideoCapture(0)  # Open the camera when the page is initialized
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_feed)
        self.timer.start(10)  # Update the feed every 10ms
        self.success_alert_shown = False  # Track if success alert is shown

    def init_ui(self):
        self.setWindowTitle("Add New Person")
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()

        # Camera feed
        self.video_feed = QLabel(self)
        self.video_feed.setFixedSize(640, 480)
        self.video_feed.setStyleSheet("background-color: #000;")  # Initially black screen
        layout.addWidget(self.video_feed, alignment=Qt.AlignCenter)

        # Counter label
        self.counter_label = QLabel("Captured: 0/50", self)
        self.counter_label.setStyleSheet("color: white; background-color: black; padding: 5px; font-size: 16px;")
        self.counter_label.setFixedSize(120, 30)
        self.counter_label.move(520, 20)  # Top-right corner of the camera feed
        self.counter_label.setVisible(True)

        # Input fields
        self.student_id_input = QLineEdit()
        self.student_id_input.setPlaceholderText("Enter Student ID")
        layout.addWidget(self.student_id_input, alignment=Qt.AlignCenter)

        self.student_name_input = QLineEdit()
        self.student_name_input.setPlaceholderText("Enter Student Name")
        layout.addWidget(self.student_name_input, alignment=Qt.AlignCenter)

        # Buttons
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_capturing)
        self.start_button.setEnabled(False)  # Initially disabled
        layout.addWidget(self.start_button, alignment=Qt.AlignCenter)

        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close_page)
        self.close_button.setVisible(False)  # Initially hidden
        layout.addWidget(self.close_button, alignment=Qt.AlignCenter)

        # Connect input fields to validation
        self.student_id_input.textChanged.connect(self.validate_inputs)
        self.student_name_input.textChanged.connect(self.validate_inputs)

        self.setLayout(layout)

    def validate_inputs(self):
        student_id = self.student_id_input.text().strip()
        student_name = self.student_name_input.text().strip()
        self.start_button.setEnabled(bool(student_id and student_name))

    def start_capturing(self):
        self.image_count = 0
        self.capture_images = True
        self.counter_label.setText("Captured: 0")

    def update_feed(self):
        ret, frame = self.cap.read()

        if ret:
            # Calculate aspect ratio
            frame_height, frame_width, _ = frame.shape
            label_width, label_height = self.video_feed.width(), self.video_feed.height()

            # Scale frame to fit within label while maintaining aspect ratio
            aspect_ratio = frame_width / frame_height
            if label_width / label_height > aspect_ratio:
                new_height = label_height
                new_width = int(aspect_ratio * new_height)
            else:
                new_width = label_width
                new_height = int(new_width / aspect_ratio)

            resized_frame = cv2.resize(frame, (new_width, new_height))

            # Create a black canvas and center the resized frame
            canvas = np.zeros((label_height, label_width, 3), dtype=np.uint8)
            x_offset = (label_width - new_width) // 2
            y_offset = (label_height - new_height) // 2
            canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_frame

            # Convert BGR to RGB for PyQt display
            rgb_frame = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.video_feed.setPixmap(QPixmap.fromImage(qimg))

            if self.capture_images and self.image_count < 50:
                # Save the frame
                student_id = self.student_id_input.text().strip()
                student_name = self.student_name_input.text().strip()
                dir_name = f"dataset/{student_name}"
                os.makedirs(dir_name, exist_ok=True)

                if self.image_count == 0:
                    filename = f"Faces/{student_id}_{student_name}.jpg"
                else:
                    filename = f"{dir_name}/{student_id}_{student_name}_{self.image_count}.jpg"

                cv2.imwrite(filename, frame)
                self.image_count += 1
                self.counter_label.setText(f"Captured: {self.image_count}")

            if self.image_count >= 50:
                self.capture_images = False  # Stop capturing
                if not self.success_alert_shown:
                    QMessageBox.information(self, "Success", "Captured 50 images successfully!")
                    self.success_alert_shown = True
                self.start_button.setVisible(False)
                self.close_button.setVisible(True)

    def close_page(self):
        if self.cap:
            self.cap.release()
        if self.timer:
            self.timer.stop()

        # Call the encode generator function
        TrainData.generate_encodings(self.class_id)
        self.close()

    def closeEvent(self, event):
        # Ensure the camera and timer are released properly when the window is closed
        if self.cap:
            self.cap.release()
        if self.timer:
            self.timer.stop()
        event.accept()
