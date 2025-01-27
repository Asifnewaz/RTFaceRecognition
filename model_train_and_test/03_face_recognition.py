import numpy as np
import cv2
import os
from PIL import Image
import tensorflow as tf
from tensorflow.python.keras.models import load_model

import time

# Load the trained model
model = tf. keras. models. load_model('../face_recognition_model.keras')
# model = load_model('../face_recognition_model.keras')

# Load class labels
# Class labels (replace with your dataset class indices)
class_labels = {0: "Asif", 1: "Mukit", 2: "Kamrul"}

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

# Load Haar Cascade for face detection
cascPath = "haarcascade_frontalface_default.xml"  # Path to Haar Cascade XML file
faceCascade = cv2.CascadeClassifier(cascPath)

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Preprocess the face for the CNN model
        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, (64, 64))  # Resize to match the input shape of the model
        face = face.astype('float32') / 255.0  # Normalize pixel values
        face = face.reshape(1, 64, 64, 1)  # Add batch dimension and channel dimension

        # Predict the class of the face
        predictions = model.predict(face)
        label_index = np.argmax(predictions)
        confidence = np.max(predictions) * 100  # Get confidence score as percentage

        # Get the label of the prediction
        label = class_labels[label_index]

        # Display label and confidence on the frame
        text = f"{label}: {confidence:.2f}%"
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Show the frame
    cv2.imshow("Face Recognition", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
