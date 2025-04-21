import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = torch.nn.Linear(model.fc.in_features, 7)  # Change 7 to your actual number of emotions
model.load_state_dict(torch.load("emotion_detection_resnet18.pth", map_location=device))
model = model.to(device)
model.eval()

# Define transformation for test image
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Load and preprocess test image
image_path = "C:/Users/flame/Documents/Image Data set/test/angry/im16.png"
image = Image.open(image_path)
image = transform(image).unsqueeze(0).to(device)

# Make prediction
with torch.no_grad():
    output = model(image)
    predicted_class = torch.argmax(output).item()

# Print result
emotions = ["Happy", "Sad", "Angry", "Surprised", "Neutral", "Disgusted", "Fearful"]  # Adjust according to your dataset
print(f"Predicted Emotion: {emotions[predicted_class]}")
