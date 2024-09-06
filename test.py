import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch import nn
from PIL import Image

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transformations
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

data_dir = "./train"
dataset = ImageFolder(data_dir, transform=transform)


def test_model(model, image_path):
    model.eval()
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)

    return dataset.classes[predicted.item()]


def calculate_accuracy(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


# Use the latest MobileNetV2 model with updated weights
# model = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.DEFAULT)
model = torchvision.models.mobilenet_v2()

# Modify the last layer for your number of classes
num_classes = 4
model.classifier[1] = nn.Linear(model.last_channel, num_classes)

# Load the saved model
model.load_state_dict(torch.load("mobilenetv2_finetuned.pth", weights_only=True))
model.to(device)

# Create data loaders for both train and test datasets
train_loader = DataLoader(dataset, batch_size=32, shuffle=False)
test_dir = "./test"
test_dataset = ImageFolder(test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Calculate accuracy for train dataset
train_accuracy = calculate_accuracy(model, train_loader, device)
print(f"Accuracy on train set: {train_accuracy:.2f}%")

# Calculate accuracy for test dataset
test_accuracy = calculate_accuracy(model, test_loader, device)
print(f"Accuracy on test set: {test_accuracy:.2f}%")
