# Image Sorting by Face and Feature Classification

This script detects faces in images and sorts them into folders based on recognized individuals using **Face Recognition**. Additionally, images without detected faces are classified based on their scene features (e.g., "beach", "city", etc.) using a **pre-trained InceptionV3 model**.

## Features
- **Face Detection**: Uses the `face_recognition` library to detect faces in images.
- **Face Recognition**: Uses facial encodings to identify known individuals.
- **Feature Classification**: Uses the InceptionV3 model from TensorFlow to classify images by their scene (e.g., "beach", "urban").
- **Folder Organization**: Automatically creates folders for each recognized person or scene and moves the original images into these folders.
- **Efficient Handling**: Handles large numbers of images, and skips images with no faces.

## Requirements
Make sure you have the following libraries installed:

- **TensorFlow**: `pip install tensorflow`
- **Face Recognition**: `pip install face_recognition`
- **Pillow (PIL)**: `pip install Pillow`
- **NumPy**: `pip install numpy`

## How to Use

1. Clone the repository or download the script files to your local machine.
   
2. Prepare the following folders:
   - `images/`: This folder should contain the images you want to sort.
   - `sorted_images/`: The folder where the sorted images will be moved. This folder will be created automatically if it doesn't exist.

3. Run the script:

   ```bash
   python sort_images.py
