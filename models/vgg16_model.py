import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os

class VGG16Custom(nn.Module):
    """
    VGG-like Convolutional Neural Network Architecture.

    Type: Deep Convolutional Neural Network (CNN) for image classification.
    Functionality: Hierarchical feature extraction from images, followed by classification.
    Speciality: Uses small 3x3 convolutional filters throughout for efficient, deep feature learning.
               All 22 layers are fully trainable, ensuring complete visibility and fine-tuning capability.
    """
    def __init__(self, num_classes):
        super(VGG16Custom, self).__init__()
        self.features = nn.Sequential(
            # Block 1: Low-level feature extraction (edges, colors, basic textures)
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Reduces spatial size by half, adds translation invariance
            # Block 2: Mid-level features (shapes, patterns)
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Block 3: Higher-level features (object parts)
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Block 4: Complex patterns (combinations of parts)
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Block 5: High-level semantic features (full objects, scenes)
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Flatten: Convert 2D feature maps to 1D vector for classification
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            # Fully connected layers for classification
            nn.Linear(25088, 4096), nn.ReLU(),  # Learns complex combinations of features
            nn.Linear(4096, 4096), nn.ReLU(),   # Further refines representations
            nn.Linear(4096, num_classes)        # Output layer: class probabilities
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_dataset(dataset_path):
    print(f"Attempting to load dataset from: {dataset_path}")
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path '{dataset_path}' does not exist.")
        raise FileNotFoundError(f"Dataset path '{dataset_path}' does not exist.")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, full_dataset.classes

def train_model(model, train_loader, test_loader, class_names, num_epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Evaluate on test set
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        test_loss /= len(test_loader)
        test_acc = 100 * correct / total
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    return train_losses, test_losses, train_accs, test_accs

def train_vgg16(dataset_path, num_epochs=5):
    train_loader, test_loader, class_names = load_dataset(dataset_path)
    num_classes = len(class_names)

    vgg16 = models.vgg16(weights='IMAGENET1K_V1')
    vgg16.classifier[6] = nn.Linear(4096, num_classes)

    # Unfreeze all layers for fine-tuning
    for param in vgg16.parameters():
        param.requires_grad = True

    vgg16 = vgg16.to(device)
    train_losses, test_losses, train_accs, test_accs = train_model(vgg16, train_loader, test_loader, class_names, num_epochs)
    os.makedirs('trained_models', exist_ok=True)
    torch.save(vgg16.state_dict(), os.path.join('trained_models', 'vgg16_model.pth'))
    print("VGG-16 model saved.")
    return train_losses, test_losses, train_accs, test_accs

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
        num_epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    else:
        dataset_path = "dataset\TRAIN"  # Adjust path as needed
        num_epochs = 5
    train_vgg16(dataset_path, num_epochs)