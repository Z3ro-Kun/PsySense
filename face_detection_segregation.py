import os
import cv2
import dlib
import shutil

# Paths to the dataset folders (adjust these as needed)
base_dir = "C:/Users/flame/Documents/EmoSet-118k"
images_dir = os.path.join(base_dir, "image")
annotations_dir = os.path.join(base_dir, "annotations")  # If needed later
output_dir = os.path.join(base_dir, "subset_faces")

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Initialize dlib's face detector
detector = dlib.get_frontal_face_detector()

# Process each emotion folder
for emotion in os.listdir(images_dir):
    emotion_folder = os.path.join(images_dir, emotion)
    output_emotion_folder = os.path.join(output_dir, emotion)
    if not os.path.exists(output_emotion_folder):
        os.makedirs(output_emotion_folder)

    # Loop over images in the emotion folder
    for img_file in os.listdir(emotion_folder):
        img_path = os.path.join(emotion_folder, img_file)
        # Read the image using OpenCV
        image = cv2.imread(img_path)
        if image is None:
            continue  # Skip files that cannot be read

        # Convert image to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = detector(gray)

        # If at least one face is detected, copy the image
        if len(faces) > 0:
            shutil.copy(img_path, os.path.join(output_emotion_folder, img_file))
            print(f"Copied: {img_path} to {output_emotion_folder}")
        else:
            print(f"No face detected in: {img_path}")

print("Subset creation complete!")
