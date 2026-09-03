import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms

from model import CNN


# CIFAR-10 classes
classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]


# Load model
model = CNN()
model.load_state_dict(torch.load("cnn_cifar10.pth", map_location="cpu"))
model.eval()


# Same preprocessing used during training
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])


# Streamlit UI
st.title("🖼️ CIFAR-10 Image Classifier")
st.write("Upload an image and let the CNN predict its class.")


uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        width=300
    )

    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    predicted_class = classes[predicted.item()]

    st.success(f"Prediction: **{predicted_class}**")
    st.info(f"Confidence: **{confidence.item() * 100:.2f}%**")