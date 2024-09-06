import torch
import torchvision
from torchvision import transforms
from torch import nn
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

model = torchvision.models.mobilenet_v2(weights=None)
num_classes = 4
model.classifier[1] = nn.Linear(model.last_channel, num_classes)
model.load_state_dict(torch.load("mobilenetv2_finetuned.pth", map_location=device))
model.to(device)
model.eval()

class_names = [
    "Dark",
    "Green",
    "Light",
    "Medium",
]


def classify_image(image_path):
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    # Perform the classification
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_class = torch.argmax(probabilities).item()

    # Get the class name and probability
    class_name = class_names[predicted_class]
    probability = probabilities[predicted_class].item()

    return class_name, probability


# Example usage
image_path = "path/to/your/image.jpg"  # Replace with the path to your image
predicted_class, confidence = classify_image(image_path)

print(f"Predicted class: {predicted_class}")
print(f"Confidence: {confidence:.2%}")
