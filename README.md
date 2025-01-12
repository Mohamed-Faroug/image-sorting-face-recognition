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
   python sorted_faces_&_classify_image_v1.py
The script will:
Detect faces and move the images into corresponding folders by person.
If no faces are detected, it will classify the image based on its feature (e.g., "beach", "city", etc.) and move it into a corresponding folder.
How It Works
The script uses Face Recognition to detect and recognize faces in the images in the images/ folder.
If a face is detected, the script compares it against known faces and moves the image to the respective person's folder.
If no faces are detected, the script classifies the image using the InceptionV3 model (pre-trained on ImageNet) and moves the image to a folder corresponding to the scene (e.g., "beach", "city").
Folder Structure
After running the script, the images will be sorted into the following folder structure:

    ```sorted_images/
       ├── Person_1/
       ├── Person_2/
       ├── beach/
       ├── city/
       ├── ...
Error Handling
Images with no faces are automatically classified by scene and moved to the appropriate folder.
Errors during image processing (e.g., corrupt images) are caught, and the script moves on to the next image.
Contributing
Feel free to fork the repository, make changes, and submit a pull request. If you have any suggestions or improvements, feel free to open an issue.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
face_recognition: A library built on top of dlib for face recognition.
TensorFlow & InceptionV3: Used for feature classification of images.
