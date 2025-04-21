import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

def main():
    # ------------------------------
    # Paths for your dataset
    # ------------------------------
    train_dir = "C:/Users/flame/Documents/Image Data set/train"
    test_dir = "C:/Users/flame/Documents/Image Data set/test"

    # ------------------------------
    # Data Transformations with Augmentation for 48x48 Images
    # ------------------------------
    transform_train = transforms.Compose([
        transforms.Resize((48, 48)),  # Changed from (256, 256)
        transforms.RandomResizedCrop(48, scale=(0.8, 1.0)),  # Changed crop size from 224 to 48
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.Resize((48, 48)),  # Changed from (256, 256)
        transforms.CenterCrop(48),  # Changed crop size from 224 to 48
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # ------------------------------
    # Dataset Loading
    # ------------------------------
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform_train)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    # ------------------------------
    # Model Definition: Pretrained ResNet18 with Custom Classification Head
    # ------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, len(train_dataset.classes))
    )
    model = model.to(device)

    # ------------------------------
    # Loss, Optimizer, and Learning Rate Scheduler
    # ------------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # ------------------------------
    # Training Loop with Logging and Test Evaluation
    # ------------------------------
    num_epochs = 20
    train_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Evaluate on test set
        model.eval()
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total_test += labels.size(0)
                correct_test += predicted.eq(labels).sum().item()
        test_acc = 100 * correct_test / total_test
        test_accuracies.append(test_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")
        scheduler.step()

    # ------------------------------
    # Save the Trained Model
    # ------------------------------
    torch.save(model.state_dict(), "emotion_model.pth")
    print("Model training complete and saved as emotion_model.pth")

    # ------------------------------
    # Plot Training and Test Accuracy Curves
    # ------------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs+1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, num_epochs+1), test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Test Accuracy')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # For Windows multiprocessing safety
    from multiprocessing import freeze_support
    freeze_support()
    main()
