import torch
import torchvision
from torchvision import transforms
from torch import nn
from PIL import Image
import gradio as gr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


model = torchvision.models.mobilenet_v2(weights=None)
num_classes = 4  # Update this to match your number of classes
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


def classify_image(image):
    if image is None:
        return {class_name: 0.0 for class_name in class_names}

    # Convert the image to a PIL Image if it's not already
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image.astype("uint8"), "RGB")

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_class = torch.argmax(probabilities).item()

    return {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}

iface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(),
    outputs=gr.Label(num_top_classes=4),
    title="Coffee Roast Classification",
    description="Upload an image of coffee beans to classify the roast level.",
    examples=[
        ["examples/dark.png"],
        ["examples/green.png"],
        ["examples/light.png"],
        ["examples/medium.png"],
    ]
)

iface.launch()
