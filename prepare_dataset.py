from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Transform (resize images)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load dataset
train_data = datasets.ImageFolder("dataset/train", transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# Print classes
print("Classes:", train_data.classes)
print("Total images:", len(train_data))
print(train_data.class_to_idx)