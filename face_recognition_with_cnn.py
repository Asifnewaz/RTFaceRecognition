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
        for i in range(5, 0, -1):
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

        self.model_path = "pretrained_face_model.h5"
        if not os.path.exists(self.model_path):
            self.train_model()

        self.model = self.load_model()
        self.class_id = ""  # To store class ID
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
        self.countdown_label.setText(str(value))

    def start_video_feed(self):
        self.stack.setCurrentWidget(self.video_feed_page)

        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_feed)
        self.timer.start(10)

    def train_model(self):
        try:
            data = []
            labels = []
            image_dir = "Images"  # Folder containing training images
            categories = os.listdir(image_dir)
            categories = [path for path in categories if path != '.DS_Store']

            for idx, category in enumerate(categories):
                category_path = os.path.join(image_dir, category)
                for img_file in os.listdir(category_path):
                    img_path = os.path.join(category_path, img_file)
                    img = cv2.imread(img_path)
                    img = cv2.resize(img, (64, 64))  # Resize to 64x64
                    data.append(img)
                    labels.append(idx)

            data = np.array(data) / 255.0  # Normalize data
            labels = to_categorical(labels)  # One-hot encode labels

            # Split data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

            # Build CNN model
            model = Sequential([
                Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
                MaxPooling2D((2, 2)),
                Conv2D(64, (3, 3), activation='relu'),
                MaxPooling2D((2, 2)),
                Flatten(),
                Dense(128, activation='relu'),
                Dense(len(categories), activation='softmax')
            ])

            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            # Train model
            model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32)

            # Save trained model
            model.save(self.model_path)
            print("Model trained and saved successfully.")
        except Exception as e:
            print(f"Failed to train model: {e}")
            exit()

    def load_model(self):
        try:
            model = load_model(self.model_path, compile=False)
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            return model
        except Exception as e:
            print(f"Failed to load model: {e}")
            exit()

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
                self.id_label.setText(f"ID: {result}")
                self.time_label.setText(f"Time: {time.strftime('%H:%M:%S')}")
                self.loader_label.setVisible(False)

                # Show success message
                msg = QMessageBox()
                msg.setWindowTitle("Recognition Success")
                msg.setText("User recognized successfully!")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()

        # Resume processing
        self.processing = False
        self.pause_processing = False

    def process_frame(self, frame):
        # Preprocess frame for model input
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            return "Unknown"

        for (x, y, w, h) in faces:
            face = frame[y:y + h, x:x + w]
            face = cv2.resize(face, (64, 64))  # Assuming the model expects 64x64 input size
            face = np.expand_dims(face, axis=0) / 255.0

            # Predict using the model
            prediction = self.model.predict(face)
            class_index = np.argmax(prediction)
            return f"Class {class_index}"

        return "Unknown"

    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication([])
    window = FaceRecognitionApp()
    window.show()
    app.exec_()

# Step 2
# import cv2
# import numpy as np
# from keras.src.models import Sequential
# from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense
# from keras.src.saving import load_model
# from keras.src.utils import to_categorical
# import cv2
# import numpy as np
# from sklearn.model_selection import train_test_split
# from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QHBoxLayout, QMessageBox
# from PyQt5.QtGui import QImage, QPixmap
# from PyQt5.QtCore import QTimer
# import os
# import time
#
# class FaceRecognitionApp(QWidget):
#     def __init__(self):
#         super().__init__()
#
#         # Check and train model if not exists
#         self.model_path = "pretrained_face_model.h5"
#         if not os.path.exists(self.model_path):
#             self.train_model()
#
#         # Load pretrained CNN model
#         self.model = self.load_model()
#
#         # Initialize GUI
#         self.setWindowTitle("Face Recognition App")
#         self.setGeometry(100, 100, 800, 600)
#
#         # Layouts
#         main_layout = QHBoxLayout()
#         video_layout = QVBoxLayout()
#         info_layout = QVBoxLayout()
#
#         # Video Feed Label
#         self.video_feed = QLabel(self)
#         self.video_feed.setFixedSize(640, 480)
#         video_layout.addWidget(self.video_feed)
#
#         # ID Label
#         self.id_label = QLabel("ID: ")
#         self.id_label.setStyleSheet("font-size: 16px;")
#         info_layout.addWidget(self.id_label)
#
#         # Time Label
#         self.time_label = QLabel("Time: ")
#         self.time_label.setStyleSheet("font-size: 16px;")
#         info_layout.addWidget(self.time_label)
#
#         # Loader Label
#         self.loader_label = QLabel("Processing...")
#         self.loader_label.setStyleSheet("font-size: 16px; color: blue;")
#         self.loader_label.setVisible(False)
#         info_layout.addWidget(self.loader_label)
#
#         # Combine layouts
#         main_layout.addLayout(video_layout)
#         main_layout.addLayout(info_layout)
#         self.setLayout(main_layout)
#
#         # Capture video
#         self.cap = cv2.VideoCapture(0)
#
#         # Timer for updating feed
#         self.timer = QTimer()
#         self.timer.timeout.connect(self.update_feed)
#         self.timer.start(10)
#
#         self.processing = False  # To track if processing is ongoing
#         self.pause_processing = False
#         self.detected_faces = set()  # To store IDs of detected faces
#
#     def train_model(self):
#         try:
#             data = []
#             labels = []
#             image_dir = "images"  # Folder containing training images
#             categories = os.listdir(image_dir)
#
#             for idx, category in enumerate(categories):
#                 category_path = os.path.join(image_dir, category)
#                 for img_file in os.listdir(category_path):
#                     img_path = os.path.join(category_path, img_file)
#                     img = cv2.imread(img_path)
#                     img = cv2.resize(img, (64, 64))  # Resize to 64x64
#                     data.append(img)
#                     labels.append(idx)
#
#             data = np.array(data) / 255.0  # Normalize data
#             labels = to_categorical(labels)  # One-hot encode labels
#
#             # Split data into train and test sets
#             X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
#
#             # Build CNN model
#             model = Sequential([
#                 Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
#                 MaxPooling2D((2, 2)),
#                 Conv2D(64, (3, 3), activation='relu'),
#                 MaxPooling2D((2, 2)),
#                 Flatten(),
#                 Dense(128, activation='relu'),
#                 Dense(len(categories), activation='softmax')
#             ])
#
#             model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#
#             # Train model
#             model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32)
#
#             # Save trained model
#             model.save(self.model_path)
#             print("Model trained and saved successfully.")
#         except Exception as e:
#             print(f"Failed to train model: {e}")
#             exit()
#
#     def load_model(self):
#         try:
#             model = load_model(self.model_path, compile=False)
#             model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#             return model
#         except Exception as e:
#             print(f"Failed to load model: {e}")
#             exit()
#
#     def update_feed(self):
#
#         ret, frame = self.cap.read()
#
#         if ret:
#             # Resize frame to center the face
#             height, width, _ = frame.shape
#             crop_size = min(height, width)
#             start_x = (width - crop_size) // 2
#             start_y = (height - crop_size) // 2
#             frame = frame[start_y:start_y + crop_size, start_x:start_x + crop_size]
#             frame = cv2.resize(frame, (640, 480))
#
#             # Update video feed
#             rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             h, w, ch = rgb_frame.shape
#             bytes_per_line = ch * w
#             qimg = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
#             self.video_feed.setPixmap(QPixmap.fromImage(qimg))
#
#             if self.pause_processing:
#                 return
#
#             # If not processing, trigger detection
#             if not self.processing:
#                 self.processing = True
#                 self.loader_label.setVisible(True)
#                 QTimer.singleShot(500, lambda: self.handle_detection(frame))
#
#     def handle_detection(self, frame):
#         self.pause_processing = True
#         result = self.process_frame(frame)
#
#         if result == "Unknown":
#             # Clear labels if no face is detected
#             self.id_label.setText("ID: ")
#             self.time_label.setText("Time: ")
#             self.loader_label.setVisible(False)
#         else:
#             if result in self.detected_faces:
#                 # Show message if attendance is already saved
#                 msg = QMessageBox()
#                 msg.setWindowTitle("Attendance Saved")
#                 msg.setText("User already marked present.")
#                 msg.setStandardButtons(QMessageBox.Ok)
#                 msg.exec_()
#             else:
#                 # Save attendance and update labels
#                 self.detected_faces.add(result)
#                 self.id_label.setText(f"ID: {result}")
#                 self.time_label.setText(f"Time: {time.strftime('%H:%M:%S')}")
#                 self.loader_label.setVisible(False)
#
#                 # Show success message
#                 msg = QMessageBox()
#                 msg.setWindowTitle("Recognition Success")
#                 msg.setText("User recognized successfully!")
#                 msg.setStandardButtons(QMessageBox.Ok)
#                 msg.exec_()
#
#         # Resume processing
#         self.processing = False
#         self.pause_processing = False
#
#     def process_frame(self, frame):
#         # Preprocess frame for model input
#         face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#
#         if len(faces) == 0:
#             return "Unknown"
#
#         for (x, y, w, h) in faces:
#             face = frame[y:y + h, x:x + w]
#             face = cv2.resize(face, (64, 64))  # Assuming the model expects 64x64 input size
#             face = np.expand_dims(face, axis=0) / 255.0
#
#             # Predict using the model
#             prediction = self.model.predict(face)
#             class_index = np.argmax(prediction)
#             return f"Class {class_index}"
#
#         return "Unknown"
#
#     def closeEvent(self, event):
#         self.cap.release()
#         event.accept()
#
# if __name__ == "__main__":
#     app = QApplication([])
#     window = FaceRecognitionApp()
#     window.show()
#     app.exec_()




# Step 1:
# class FaceRecognitionApp:
#     def __init__(self):
#         # Check and train model if not exists
#         self.model_path = "pretrained_face_model.h5"
#         if not os.path.exists(self.model_path):
#             self.train_model()
#
#         # Load pretrained CNN model
#         self.model = self.load_model()
#
#     def train_model(self):
#         try:
#             data = []
#             labels = []
#             image_dir = "images"  # Folder containing training images
#             categories = os.listdir(image_dir)
#
#             for idx, category in enumerate(categories):
#                 category_path = os.path.join(image_dir, category)
#                 for img_file in os.listdir(category_path):
#                     img_path = os.path.join(category_path, img_file)
#                     img = cv2.imread(img_path)
#                     img = cv2.resize(img, (64, 64))  # Resize to 64x64
#                     data.append(img)
#                     labels.append(idx)
#
#             data = np.array(data) / 255.0  # Normalize data
#             labels = to_categorical(labels)  # One-hot encode labels
#
#             # Split data into train and test sets
#             X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
#
#             # Build CNN model
#             model = Sequential([
#                 Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
#                 MaxPooling2D((2, 2)),
#                 Conv2D(64, (3, 3), activation='relu'),
#                 MaxPooling2D((2, 2)),
#                 Flatten(),
#                 Dense(128, activation='relu'),
#                 Dense(len(categories), activation='softmax')
#             ])
#
#             model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#
#             # Train model
#             model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32)
#
#             # Save trained model
#             model.save(self.model_path)
#             print("Model trained and saved successfully.")
#         except Exception as e:
#             print(f"Failed to train model: {e}")
#             exit()
#
#     def load_model(self):
#         try:
#             model = load_model(self.model_path, compile=False)
#             model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#             return model
#         except Exception as e:
#             print(f"Failed to load model: {e}")
#             exit()
#
#     def process_image(self, file_path):
#         try:
#             image = cv2.imread(file_path)
#             self.process_frame(image)
#         except Exception as e:
#             print(f"Failed to process image: {e}")
#
#     def process_frame(self, frame):
#         # Preprocess frame for model input
#         face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#
#         if len(faces) == 0:
#             print("Result: No face detected")
#             return
#
#         for (x, y, w, h) in faces:
#             face = frame[y:y + h, x:x + w]
#             face = cv2.resize(face, (64, 64))  # Assuming the model expects 64x64 input size
#             face = np.expand_dims(face, axis=0) / 255.0
#
#             # Predict using the model
#             prediction = self.model.predict(face)
#             class_index = np.argmax(prediction)
#
#             # Update result
#             print(f"Result: Class {class_index}")
#
# if __name__ == "__main__":
#     app = FaceRecognitionApp()
#
#     # Example usage:
#     app.process_image("images/person2/1547470.png")
#     # Use external scripts to integrate or call this class for GUI if needed.
