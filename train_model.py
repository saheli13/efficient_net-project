import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Image transform
transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load dataset
train_dataset = datasets.ImageFolder(
    "dataset/train",
    transform=transform
)

val_dataset = datasets.ImageFolder(
    "dataset/val",
    transform=transform
)

# Check class labels
print(train_dataset.class_to_idx)

# Data loader
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16
)

# Load EfficientNet
model = models.efficientnet_b0(pretrained=True)

# Change output layer
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    2
)

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

# Training
epochs = 4

for epoch in range(epochs):

    model.train()

    running_loss = 0

    for images, labels in train_loader:

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {running_loss}")

# Save model
torch.save(
    model.state_dict(),
    "efficient_model.pth"
)

print("Model Saved")

# ---------------- EVALUATION ----------------

model.eval()

all_labels = []
all_predictions = []

with torch.no_grad():

    for images, labels in val_loader:

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        all_labels.extend(labels.numpy())
        all_predictions.extend(predicted.numpy())

# Metrics
accuracy = accuracy_score(all_labels, all_predictions)

precision = precision_score(all_labels, all_predictions)

recall = recall_score(all_labels, all_predictions)

f1 = f1_score(all_labels, all_predictions)

cm = confusion_matrix(all_labels, all_predictions)

# Print metrics
print(f"\nAccuracy: {accuracy * 100:.2f}%")

print(f"Precision: {precision * 100:.2f}%")

print(f"Recall: {recall * 100:.2f}%")

print(f"F1 Score: {f1 * 100:.2f}%")

print("\nConfusion Matrix:")

print(cm)