import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.src.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizer_v1 import Adam


# Load images
def load_images_from_folder(folder_path, img_size=(64, 64)):
    images = []
    labels = []
    label_map = {}
    current_label = 0

    for person_name in os.listdir(folder_path):
        person_folder = os.path.join(folder_path, person_name)
        if os.path.isdir(person_folder):
            label_map[current_label] = person_name
            for filename in os.listdir(person_folder):
                img_path = os.path.join(person_folder, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img_resized = cv2.resize(img, img_size)
                    images.append(img_resized)
                    labels.append(current_label)
            current_label += 1

    return np.array(images), np.array(labels), label_map


# Load dataset
folder_path = 'Images'
X, y, label_map = load_images_from_folder(folder_path)

# Preprocess data
X = X / 255.0
X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
y = to_categorical(y, num_classes=len(label_map))

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(label_map), activation='softmax')
])

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Evaluate model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")


# Predict function
def predict_face(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (64, 64))
    img_normalized = img_resized / 255.0
    img_reshaped = img_normalized.reshape(1, 64, 64, 1)
    prediction = model.predict(img_reshaped)
    predicted_label = np.argmax(prediction)
    person_name = label_map[predicted_label]

    plt.imshow(img_resized, cmap='gray')
    plt.title(f"Predicted: {person_name}")
    plt.axis('off')
    plt.show()


# Test prediction
predict_face('Images/person1/1446896.png')