import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

device = "cpu"

model = models.densenet121(weights=None)
model.classifier = nn.Linear(1024, 2)

checkpoint = torch.load(
    "faceproof_densenet.pth",
    map_location=device
)

model.load_state_dict(checkpoint["model"])
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

classes = ["Fake", "Real"]

def predict(img):

    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        probs = torch.softmax(output, dim=1)[0]

    return {
        classes[0]: float(probs[0]),
        classes[1]: float(probs[1])
    }

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=2),
    title="DeepFake Image Detection",
    description="DenseNet121 Based Deepfake Detection System"
)

demo.launch()
