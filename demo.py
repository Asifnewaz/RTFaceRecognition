import pickle
import sys
import cv2
import cvzone
import numpy as np
import face_recognition
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt


class FaceRecognitionApp(QMainWindow):
    # Global variables for window and camera feed dimensions
    WINDOW_WIDTH = 1084
    WINDOW_HEIGHT = 800
    VIDEO_FEED_WIDTH = 640
    VIDEO_FEED_HEIGHT = 400
    TOP_PADDING = 10
    LEFT_PADDING = 10
    PADDING = 10
    RIGHT_IMAGE_WIDTH = 414
    RIGHT_IMAGE_HEIGHT = 633
    encodeListKnown = []
    studentIds = []
    id = -1

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Face Recognition")
        self.setGeometry(0, 0, self.WINDOW_WIDTH, self.WINDOW_HEIGHT)

        # Set Nordic-inspired background color
        self.setStyleSheet("background-color: #DCE2E6;")

        # Central widget
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        # QLabel to display the video feed
        self.video_label = QLabel(self.central_widget)
        self.video_label.setGeometry(self.LEFT_PADDING, self.TOP_PADDING, self.VIDEO_FEED_WIDTH, self.VIDEO_FEED_HEIGHT)  # Positioned at (5, 5)
        self.video_label.setStyleSheet("background-color: #AEBFCF; border: 0px solid #7A8D99;")

        # QLabel to display the image
        self.image_label = QLabel(self.central_widget)
        image_x_position = self.LEFT_PADDING + self.VIDEO_FEED_WIDTH + self.PADDING   # Position the image label to the right of the video feed
        self.image_label.setGeometry(image_x_position, self.TOP_PADDING, self.RIGHT_IMAGE_WIDTH, self.RIGHT_IMAGE_HEIGHT)
        self.image_label.setStyleSheet("background-color: #C0D6DF; border: 0px solid #7A8D99;")

        # Load and display the image
        self.load_image("Resources/Modes/1.png")  # Replace with your image path

        # Load the encoding file
        print("Loading Encode File ...")
        file = open('EncodeFile.p', 'rb')
        encodeListKnownWithIds = pickle.load(file)
        file.close()
        self.encodeListKnown, self.studentIds = encodeListKnownWithIds

        print(self.studentIds)

        # Initialize the camera
        self.cap = cv2.VideoCapture(0)  # 0 for default camera

        # Timer to update the video feed
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Refresh rate: 30 ms

    def load_image(self, image_path):
        # Load the image and display it in the QLabel
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            pixmap = pixmap.scaled(self.RIGHT_IMAGE_WIDTH, self.RIGHT_IMAGE_HEIGHT, Qt.KeepAspectRatio)
            self.image_label.setPixmap(pixmap)
        else:
            self.image_label.setText("Image not found")
            self.image_label.setAlignment(Qt.AlignCenter)

    def update_frame(self):
        ret, image = self.cap.read()
        if ret:
            # Resize the frame to fit the QLabel dimensions
            # image_resized = cv2.resize(image, (self.VIDEO_FEED_WIDTH, self.VIDEO_FEED_HEIGHT))
            image_resized = cv2.resize(image, (160, 120))
            # Process the frame for face recognition
            frame_processed = self.process_frame(image_resized)

            # Convert the frame to QImage and display it
            image = self.convert_cv_qt(frame_processed)
            self.video_label.setPixmap(QPixmap.fromImage(image))

    def process_frame(self, image):
        # Convert BGR (OpenCV) to RGB (face_recognition library)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # # Find all face locations in the frame
        face_locations = face_recognition.face_locations(rgb_image, model='cnn')
        encode_cur_frame = face_recognition.face_encodings(rgb_image, face_locations)

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
                    text = "Face Detected"
                    font_scale = 0.6
                    font_color = (0, 255, 0)  # Green color
                    font_thickness = 1
                    font = cv2.FONT_HERSHEY_SIMPLEX

                    # Put the text on the frame without background
                    cv2.putText(image, text, (left, top - 10), font, font_scale, font_color, font_thickness,
                                cv2.LINE_AA)

                print(self.id)

        return image

    def convert_cv_qt(self, frame):
        # Convert the frame to QImage
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        return QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

    def closeEvent(self, event):
        # Release the camera when the window is closed
        self.cap.release()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FaceRecognitionApp()
    window.show()
    sys.exit(app.exec_())