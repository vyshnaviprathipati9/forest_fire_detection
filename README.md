# Forest Fire Detection Using CNN

## Project Overview

Forest fires can cause significant damage to forests, wildlife, the environment, and human lives. Early detection of forest fires is important to help reduce their impact and support timely response.

This project, **Forest Fire Detection Using CNN**, aims to detect fire in forest images using a Convolutional Neural Network (CNN). The system provides image-based prediction through a Streamlit web application and live camera detection using OpenCV.

The goal is to explore how deep learning and computer vision can support forest fire detection and contribute to the development of early warning systems.

<<<<<<< HEAD
=======

>>>>>>> 64a0460 (commit)
## Features
- Fire and no-fire image classification
- CNN model trained using labelled images
- Image upload and prediction through Streamlit
- Live camera detection using OpenCV
- Prediction confidence percentage
- Saved trained model for reuse

## Technologies Used
- Python
- TensorFlow and Keras
- Convolutional Neural Network (CNN)
- Streamlit
- NumPy
- Pillow
- OpenCV
- Matplotlib

## Project Structure
```text
forest_fire_detection/
├── models/
│   └── forest_fire_cnn.keras
├── screenshots/
│   ├── forest_fire_prediction.png
│   └── forest_no_fire_prediction.png
├── app.py
├── live_camera.py
├── train_model.py
├── requirements.txt
└── README.md

```

## Installation

1. Install Python and open a terminal in the project folder.
2. Install the required packages:

```bash
pip install -r requirements.txt
```

## How to Run

### 1. Streamlit Web Application

Run:

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal.

Upload an image to view the predicted class and model confidence.

### 2. Live Camera Detection

Run:

```bash
python live_camera.py
```

Allow camera access if prompted. Press **Q** to close the camera window.

### 3. Train the Model Again (Optional)

Run:

```bash
python train_model.py
```

Ensure the labelled training dataset is available at the path configured in the training script. Training may overwrite the saved model.

## Model Details
The CNN accepts RGB images resized to 128 × 128 pixels. It uses convolutional and pooling layers to learn image features, followed by dense layers for classification. A sigmoid output produces a score used to classify an image as fire or no fire.

## Training Results
In one training run, the model achieved:
- Training accuracy: approximately 97.49%
- Validation accuracy: approximately 94.48%

These figures describe that training run and do not guarantee performance on new images.

## Testing and Limitations

The model was tested using fire and no-fire images through the Streamlit application and live camera interface. It produced correct predictions for the images tested.

Performance may vary depending on image quality, lighting, environmental conditions, and differences between training images and new images. The displayed confidence score represents the model's prediction score and does not guarantee correctness.


## Future Improvements
- Test on a larger, separate dataset of unseen images.
- Improve the diversity and quality of the training data.
- Evaluate precision, recall, and the confusion matrix.
- Investigate false-negative predictions.
- Improve the interface and model reliability.

## Project Screenshots

### Fire Detection
![Forest Fire Detection](screenshots/forest_fire_prediction.png)

### No-Fire Detection
![No-Fire Detection](screenshots/forest_no_fire_prediction.png)
