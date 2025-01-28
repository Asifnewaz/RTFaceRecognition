import cv2
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras import callbacks
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.src.models import Sequential


class TrainData:
    @staticmethod
    def train_model(classID):
        TrainData.train_cnn_model(classID)

        import cv2
        import face_recognition
        import pickle
        import os

        folderPath = 'Faces'
        pathList = os.listdir(folderPath)
        pathList = [path for path in pathList if path != '.DS_Store']  # Filter non-image files

        print(pathList)
        imgList = []
        studentIds = []
        for path in pathList:
            imgList.append(cv2.imread(os.path.join(folderPath, path)))
            studentIds.append(os.path.splitext(path)[0])

        print(studentIds)

        def findEncodings(imagesList):
            encodeList = []
            for img in imagesList:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                encode = face_recognition.face_encodings(img)[0]
                encodeList.append(encode)

            return encodeList

        encodeListKnown = findEncodings(imgList)
        encodeListKnownWithIds = [encodeListKnown, studentIds]

        file = open(f"model_train_and_test/{classID}_model.p", 'wb')
        pickle.dump(encodeListKnownWithIds, file)
        file.close()
        print("File Saved")

    @staticmethod
    def train_cnn_model(classID):
        # Path for face image database
        path = 'dataset'

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

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
        checkpoint = callbacks.ModelCheckpoint('face_recognition_model.keras', save_best_only=True, verbose=1)
        history = model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=10,
            callbacks=[checkpoint]
        )

        print("\nTraining Complete. Model saved as 'face_recognition_model.keras'")