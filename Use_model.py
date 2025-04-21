import os
from deepface import DeepFace

# Set the path to your dataset directory (organized by emotion folders)
dataset_dir = "C:/Users/flame/Documents/Image Data set/test"  # change to your dataset path

# Create a dictionary to store predictions per emotion folder
predictions = {}

# Iterate through each emotion folder in the dataset directory
for emotion_folder in os.listdir(dataset_dir):
    folder_path = os.path.join(dataset_dir, emotion_folder)
    # Ensure we are only processing directories
    if not os.path.isdir(folder_path):
        continue

    predictions[emotion_folder] = []
    print(f"\nProcessing folder: {emotion_folder}")

    # Iterate over each image file in the folder
    for img_file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_file)
        try:
            # Analyze the image for emotion
            result = DeepFace.analyze(img_path=img_path, actions=['emotion'], enforce_detection=False)
            # DeepFace.analyze returns a list of dictionaries if more than one face is found;
            # we assume one face per image and take the first result.
            predicted_emotion = result[0]["dominant_emotion"]
            predictions[emotion_folder].append(predicted_emotion)
            print(f"Image: {img_file} | Predicted: {predicted_emotion}")
        except Exception as e:
            print(f"Error processing {img_file}: {e}")

# (Optional) Summarize the predictions for each folder
print("\nSummary of Predictions:")
for emotion_folder, preds in predictions.items():
    if preds:
        avg_accuracy = sum([1 for pred in preds if pred.lower() == emotion_folder.lower()]) / len(preds) * 100
        print(f"{emotion_folder}: {len(preds)} images, Matching predictions: {avg_accuracy:.2f}%")
    else:
        print(f"{emotion_folder}: No predictions.")

