# RTFaceRecognition

## Overview

RTFaceRecognition is a real-time face recognition system built using Convolutional Neural Networks (CNN). This project captures images, trains a CNN model, and performs real-time face recognition using a webcam.

## Features

- **Image Capture**: Capture 70 snapshots of a user's face and save them in the 'dataset' folder.
- **Model Training**: Train a CNN model with the captured images and save the trained model weights.
- **Real-Time Recognition**: Use a webcam to perform real-time face recognition.

## Getting Started

### Prerequisites

- Python 3.x
- Required Python libraries (install using `requirements.txt`)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Asifnewaz/RTFaceRecognition.git
cd RTFaceRecognition
```

Install the required dependencies:
```bash
pip install -r requirements.txt
```
### Usage
Capture Images:
Run the following script to capture 70 images of your face:
```bash
python model_train_and_test/01_face_dataset.py
```

### Contributing
Contributions are welcome! Please open an issue or submit a pull request.

### License
This project is licensed under the MIT License.
