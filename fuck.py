from deepface import DeepFace

# Path to your input image
img_path = "C:/Users/flame/picture/Fuck2.png"

# Analyze the image for emotion detection
analysis = DeepFace.analyze(img_path=img_path, actions=['emotion'])

# Extract the dominant emotion and its confidence score
dominant_emotion = analysis[0]['dominant_emotion']
emotion_confidences = analysis[0]['emotion']
confidence_score = emotion_confidences[dominant_emotion]

print(f"Dominant Emotion: {dominant_emotion} ({confidence_score:.2f}%)")
print(emotion_confidences)

import math


def assess_psychological_conditions(emotion_confidences):
    """
    Assess the likelihood of psychological conditions (anxiety, depression)
    based on the emotion confidence percentages.

    Parameters:
      emotion_confidences (dict): A dictionary with emotion keys and their associated
          confidence percentages. Expected keys:
          'angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'

    Returns:
      dict: A dictionary with risk scores, probabilities, and binary flags for each condition.
    """

    # Define hypothetical weights for each emotion for anxiety and depression.
    # Positive weights mean that higher percentages contribute to higher risk,
    # negative weights reduce the risk.
    anxiety_weights = {
        'fear': 0.5,
        'angry': 0.3,
        'disgust': 0.2,
        'sad': 0.1,
        'happy': -0.4,  # Higher happiness reduces anxiety risk
        'surprise': 0.1,
        'neutral': 0.0
    }

    depression_weights = {
        'sad': 0.6,
        'neutral': 0.3,
        'fear': 0.2,
        'disgust': 0.2,
        'angry': 0.1,
        'happy': -0.5,  # Higher happiness lowers depression risk
        'surprise': 0.1
    }

    def weighted_sum(weights):
        total = 0
        for emotion, weight in weights.items():
            # Default to 0 if an emotion key is missing
            total += emotion_confidences.get(emotion, 0) * weight
        return total

    # Compute raw risk scores
    anxiety_score = weighted_sum(anxiety_weights)
    depression_score = weighted_sum(depression_weights)

    # Use logistic function to map raw scores to probabilities in range (0,1)
    def logistic(x, bias=0):
        return 1 / (1 + math.exp(-(x - bias)))

    # Adjust biases based on domain knowledge or tuning. These biases shift the sigmoid.
    anxiety_prob = logistic(anxiety_score, bias=50)  # Adjust bias as needed
    depression_prob = logistic(depression_score, bias=50)

    # Set thresholds on probabilities for binary classification (can be tuned)
    anxiety_flag = anxiety_prob > 0.5
    depression_flag = depression_prob > 0.5

    return {
        "anxiety_score": anxiety_score,
        "depression_score": depression_score,
        "anxiety_probability": anxiety_prob,
        "depression_probability": depression_prob,
        "anxiety": anxiety_flag,
        "depression": depression_flag
    }


# Example usage:
# emotion_confidences = {
#     'angry': 10.0,
#     'disgust': 5.0,
#     'fear': 20.0,
#     'happy': 50.0,
#     'sad': 10.0,
#     'surprise': 3.0,
#     'neutral': 2.0
# }

result = assess_psychological_conditions(emotion_confidences)
print("Assessment Result:")
for key, value in result.items():
    print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")

