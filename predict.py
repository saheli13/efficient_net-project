import torch
import torch.nn as nn

from torchvision import transforms, models
from PIL import Image

# Load model
model = models.efficientnet_b0(pretrained=False)

# Change output layer
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    2
)

# Load trained weights
model.load_state_dict(
    torch.load("efficient_model.pth")
)

# Evaluation mode
model.eval()

# Transform
transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Prediction function
def predict(image_path):

    # Open image
    image = Image.open(image_path).convert("RGB")

    # Transform image
    image = transform(image).unsqueeze(0)

    # Predict
    with torch.no_grad():

        outputs = model(image)

        probabilities = torch.softmax(outputs, dim=1)

        confidence = torch.max(probabilities).item() * 100

        _, predicted = torch.max(outputs, 1)

    # Print class
    if predicted.item() == 0:
        print(f"Moving Animal ({confidence:.2f}%)")
    else:
        print(f"Static Animal ({confidence:.2f}%)")

# Test image
predict("test2.jpg")