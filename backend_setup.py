import cv2
import time
import json
import sqlite3
import math
import os
import tempfile
from deepface import DeepFace
from mtcnn import MTCNN

detector = MTCNN()


def aggregate_emotions(emotion_list):
    """
    Aggregates a list of emotion confidence dictionaries by computing the average for each emotion.
    """
    if not emotion_list:
        return {}
    keys = emotion_list[0].keys()
    aggregate = {key: 0.0 for key in keys}
    for emotions in emotion_list:
        for key in keys:
            aggregate[key] += emotions.get(key, 0)
    n = len(emotion_list)
    for key in keys:
        aggregate[key] /= n
    return aggregate


def setup_database(db_name="emotion_data.db"):
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS aggregated_emotions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT,
            timestamp TEXT,
            aggregated_confidences TEXT
        )
    ''')
    conn.commit()
    return conn, c


def detect_faces(frame):
    """
    Detects faces using MTCNN and returns bounding boxes in (x, y, w, h) format.
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detections = detector.detect_faces(rgb_frame)

    boxes = []
    for detection in detections:
        x, y, w, h = detection['box']
        # Make sure coordinates are within bounds
        x, y = max(0, x), max(0, y)
        boxes.append((x, y, w, h))

    return boxes


def extract_face(frame, bbox):
    """
    Extracts the face region from the frame given a bounding box.
    """
    x, y, w, h = bbox
    return frame[y:y + h, x:x + w]


def identify_student(face_image):
    """
    Identify a student from a given face image using DeepFace's face matching.
    If no match is found, prompt the user to enter a new student ID and save a reference image.
    """
    temp_filename = os.path.join(tempfile.gettempdir(), "temp_face.jpg")
    cv2.imwrite(temp_filename, face_image)

    # Path to your student database directory (each subfolder is named with the student's ID)
    student_database_path = "C:/Users/flame/picture"  # Update this path!

    try:
        results = DeepFace.find(img_path=temp_filename, db_path=student_database_path, enforce_detection=False)
        if len(results) > 0 and not results[0].empty:
            best_match_path = results[0].iloc[0]['identity']
            student_id = os.path.basename(os.path.dirname(best_match_path))
            return student_id
    except Exception as e:
        print("Error in identify_student:", e)

    # If no match is found, prompt for a new student ID
    student_id = input("New face detected. Enter student ID: ")
    new_student_dir = os.path.join(student_database_path, student_id)
    os.makedirs(new_student_dir, exist_ok=True)
    ref_path = os.path.join(new_student_dir, f"ref_{int(time.time())}.jpg")
    cv2.imwrite(ref_path, face_image)

    if os.path.exists(temp_filename):
        os.remove(temp_filename)
    return student_id


if __name__ == '__main__':
    conn, cursor = setup_database()
    student_emotion_data = {}  # Dictionary: {student_id: [emotion_dict1, emotion_dict2, ...]}
    aggregation_interval = 60  # seconds
    interval_start = time.time()

    cap = cv2.VideoCapture(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces in the frame
        faces = detect_faces(frame)

        # Process each detected face
        for bbox in faces:
            face_image = extract_face(frame, bbox)
            student_id = identify_student(face_image)
            if student_id is None:
                continue  # Skip if still unrecognized

            try:
                result = DeepFace.analyze(face_image, actions=['emotion'], enforce_detection=False)
                emotion_dict = result[0]['emotion']  # Use the first result if multiple faces detected
            except Exception as e:
                print(f"Error processing face for {student_id}: {e}")
                continue

            if student_id not in student_emotion_data:
                student_emotion_data[student_id] = []
            student_emotion_data[student_id].append(emotion_dict)

        # Check if the aggregation interval has passed
        if time.time() - interval_start >= aggregation_interval:
            for student_id, emotion_list in student_emotion_data.items():
                if emotion_list:
                    aggregated = aggregate_emotions(emotion_list)
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    aggregated_json = json.dumps(aggregated)
                    cursor.execute("""
                        INSERT INTO aggregated_emotions (student_id, timestamp, aggregated_confidences)
                        VALUES (?, ?, ?)
                    """, (student_id, timestamp, aggregated_json))
                    conn.commit()
                    print(f"Stored aggregated data for {student_id} at {timestamp}: {aggregated}")
                    student_emotion_data[student_id] = []  # Clear buffer for next interval
            interval_start = time.time()

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    conn.close()
