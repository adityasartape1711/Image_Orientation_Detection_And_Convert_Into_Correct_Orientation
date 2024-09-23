
# Image Orientation Detection and Conversion into Correct Orientation

## Overview

**Image Orientation Detection and Conversion into Correct Orientation** is a deep learning project designed to automatically detect and correct the orientation of images. This project focuses on Marathi handwritten text images that are in various orientations, such as upside down, rotated, or skewed. The primary objective is to create a model that can accurately detect the orientation of these images and convert them into the correct orientation.

## Project Description

In this project, I have developed a Convolutional Neural Network (CNN) model to solve the problem of image orientation detection. The dataset used for training consists of images extracted from a PDF (`mangilal.pdf`) that contains Marathi handwritten text in different orientations. The images were preprocessed and labeled according to their orientation, and the CNN model was trained to classify the orientation and correct it.

The model is designed to handle a variety of common orientations such as:
- 0 degrees (upright)
- 90 degrees (clockwise)
- 180 degrees (upside down)
- 270 degrees (counterclockwise)

Once the orientation is detected, the image is then rotated back to its correct orientation.

## Key Features

- **Deep Learning Model**: A CNN architecture is employed for detecting the orientation of the images and correcting it.
- **Marathi Handwritten Text**: The dataset consists of Marathi handwritten text images extracted from a PDF file, each in varying orientations.
- **Image Processing**: The model is capable of detecting multiple orientations (0°, 90°, 180°, and 270°) and automatically converting them into the correct upright orientation.
- **Training Data**: The dataset is generated from a PDF (`mangilal.pdf`) and includes Marathi text with various orientations.

## Technologies Used

- **Python**: The project is built using Python programming language.
- **Convolutional Neural Network (CNN)**: The deep learning model used for image orientation detection.
- **TensorFlow/Keras**: For building and training the neural network.
- **OpenCV**: For image processing and manipulation.
- **NumPy & Pandas**: For data handling and preprocessing.
- **Matplotlib**: For visualizing training results and performance metrics.

## Project Structure

- `mangilal.pdf`: The PDF containing Marathi handwritten text images in various orientations.
- `notebooks/`: Jupyter notebooks for training the model and performing exploratory data analysis.
- `models/`: Saved CNN models and weights.
- `src/`: Source code for the project, including model training and orientation correction functions.
- `README.md`: Project documentation.

## How to Run the Project

1. Clone the repository.
2. Install the required dependencies listed in the `requirements.txt` file.
3. Run the training script to train the CNN model on the dataset.
4. Use the pre-trained model to predict and correct the orientation of new images.

```bash
git clone https://github.com/yourusername/image-orientation-detection.git
cd image-orientation-detection
pip install -r requirements.txt
python train_model.py
```

## Results

The CNN model achieved significant accuracy in detecting the orientation of Marathi handwritten text images and successfully corrected the orientation to its upright position.

## Future Improvements

- Expansion to detect and correct orientations of other languages' handwritten texts.
- Increase the robustness of the model by adding more challenging image rotations and distortions.
- Optimize the model for real-time image orientation detection and correction.
