import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import cv2
import collections
import time
from deepface import DeepFace

# Open webcam
cap = cv2.VideoCapture(0)

# Store last 10 predictions for stability
emotion_window = collections.deque(maxlen=10)

# Timer to process every 1.5 seconds
last_time = time.time()

# Initialize stable_emotion
stable_emotion = "Neutral"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Process emotions every 1.5 seconds
    if time.time() - last_time >= 1.5:
        try:
            # Predict emotions (avoid crash if no face detected)
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

            # If DeepFace detects a face, update emotions
            if isinstance(result, list) and len(result) > 0 and 'dominant_emotion' in result[0]:
                dominant_emotion = result[0]['dominant_emotion']

                # Store in moving window
                emotion_window.append(dominant_emotion)

                # Find most common emotion in the last 10 frames
                stable_emotion = max(set(emotion_window), key=emotion_window.count)

        except Exception as e:
            print(f"Error: {e}")  # Error handling, won't flood the console

        last_time = time.time()  # Reset timer

    # Display the stable emotion on the video
    cv2.putText(frame, f"Emotion: {stable_emotion}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Emotion Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
