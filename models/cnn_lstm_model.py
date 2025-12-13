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

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNNLSTM(nn.Module):
    def __init__(self, num_classes):
        super(CNNLSTM, self).__init__()
        self.cnn = models.mobilenet_v2(weights='IMAGENET1K_V1')
        self.cnn.classifier = nn.Identity()  # Remove the final classifier
        self.lstm = nn.LSTM(input_size=1280, hidden_size=256, num_layers=2, batch_first=True)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        # Extract features with CNN
        features = self.cnn(x)  # Shape: (batch, 512)
        # Reshape for LSTM: treat as sequence of length 1
        features = features.unsqueeze(1)  # (batch, 1, 512)
        lstm_out, _ = self.lstm(features)
        out = self.fc(lstm_out[:, -1, :])  # Take the last output
        return out

def load_dataset(dataset_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader, full_dataset.classes

def train_model(model, train_loader, test_loader, class_names, num_epochs, optimizer, start_epoch=0, train_losses=[], test_losses=[], train_accs=[], test_accs=[]):
    criterion = nn.CrossEntropyLoss()

    try:
        for epoch in range(start_epoch, start_epoch + num_epochs):
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

            print(f'Epoch {epoch+1}/{start_epoch + num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')

            # Save checkpoint after each epoch
            save_checkpoint(model, optimizer, epoch+1, train_losses, test_losses, train_accs, test_accs, 'cnn_lstm_checkpoint.pth')
    except KeyboardInterrupt:
        print("Training interrupted by user. Saving current checkpoint...")
        save_checkpoint(model, optimizer, epoch+1, train_losses, test_losses, train_accs, test_accs, 'cnn_lstm_checkpoint.pth')
        return train_losses, test_losses, train_accs, test_accs

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    return train_losses, test_losses, train_accs, test_accs

def save_checkpoint(model, optimizer, epoch, train_losses, test_losses, train_accs, test_accs, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_accs': train_accs,
        'test_accs': test_accs
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved: {filename}")

def load_checkpoint(model, optimizer, filename):
    if os.path.exists(filename):
        checkpoint = torch.load(filename, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        train_losses = checkpoint.get('train_losses', [])
        test_losses = checkpoint.get('test_losses', [])
        train_accs = checkpoint.get('train_accs', [])
        test_accs = checkpoint.get('test_accs', [])
        print(f"Loaded checkpoint from epoch {epoch}")
        return epoch, train_losses, test_losses, train_accs, test_accs
    else:
        print("No checkpoint found, starting from scratch")
        return 0, [], [], [], []

def train_cnn_lstm(dataset_path, num_epochs=5, resume=False):
    train_loader, test_loader, class_names = load_dataset(dataset_path)
    num_classes = len(class_names)

    model = CNNLSTM(num_classes)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    checkpoint_file = 'cnn_lstm_checkpoint.pth'
    start_epoch, train_losses, test_losses, train_accs, test_accs = load_checkpoint(model, optimizer, checkpoint_file) if resume else (0, [], [], [], [])

    remaining_epochs = num_epochs - start_epoch
    if remaining_epochs > 0:
        new_losses = train_model(model, train_loader, test_loader, class_names, remaining_epochs, optimizer, start_epoch, train_losses, test_losses, train_accs, test_accs)
        train_losses, test_losses, train_accs, test_accs = new_losses

    torch.save(model, 'cnn_lstm_model_complete.pth')
    print("Complete CNN-LSTM model saved.")
    return train_losses, test_losses, train_accs, test_accs

if __name__ == "__main__":
    # Build a path relative to this script to avoid depending on the current working directory
    default_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset', 'TRAIN'))
    # Also consider a path relative to the current working directory as a fallback
    cwd_path = os.path.abspath(os.path.join(os.getcwd(), 'dataset', 'TRAIN'))

    if os.path.exists(default_path):
        dataset_path = default_path
    elif os.path.exists(cwd_path):
        dataset_path = cwd_path
    else:
        raise FileNotFoundError(
            f"Dataset path not found. Tried:\n  script-relative: {default_path}\n  cwd-relative:    {cwd_path}\nPlease ensure the `dataset/TRAIN` folder exists.")

    train_cnn_lstm(dataset_path, num_epochs=2, resume=True)