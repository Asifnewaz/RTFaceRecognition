import cv2
import numpy as np
from PIL import Image
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras import callbacks
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.src.models import Sequential

# Path for face image database
path = 'dataset'

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");


def downsample_image(img):
    img = Image.fromarray(img.astype('uint8'), 'L')
    img = img.resize((32,32), Image.LANCZOS)
    return np.array(img)


# Define image dimensions and path
IMG_WIDTH, IMG_HEIGHT = 64, 64
DATASET_PATH = 'dataset'

# Prepare dataset using ImageDataGenerator
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=0.2,  # Split 20% for validation
    rotation_range=20,  # Augmentation: Rotate images
    width_shift_range=0.2,  # Augmentation: Shift width
    height_shift_range=0.2,  # Augmentation: Shift height
    zoom_range=0.2  # Augmentation: Zoom
)

train_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=32,
    color_mode='grayscale',
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=32,
    color_mode='grayscale',
    class_mode='categorical',
    subset='validation'
)

# Get the number of classes
num_classes = len(train_generator.class_indices)


# Define the CNN model
def build_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Build and summarize the model
model = build_model((IMG_WIDTH, IMG_HEIGHT, 1), num_classes)
model.summary()

# Train the model
checkpoint = callbacks.ModelCheckpoint('../face_recognition_model.keras', save_best_only=True, verbose=1)
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10,
    callbacks=[checkpoint]
)

print("\nTraining Complete. Model saved as 'face_recognition_model.keras'")