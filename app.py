import streamlit as st
import torch
import math
from torchvision import transforms
from PIL import Image
from model import MNIST_Classifier  # Import your model class

st.title("MNIST Digit Recognizer")
st.write("Upload a handwritten digit (28x28 pixels)")

# Load the trained model
model = MNIST_Classifier()
model.load_state_dict(torch.load('models/mnist_model.pth', map_location=torch.device('cpu')))
model.eval()

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    st.image(image, caption="Uploaded Image", width=100)
    
    # Preprocess and predict
    transform = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)
        confidence, prediction = torch.max(probabilities, dim=1)
        confidence = confidence.item()  # Convert to Python float
        confidence = math.floor(confidence * 10000) / 10000
        prediction = prediction.item()  # Convert to Python int
    st.write(f"Prediction: **{prediction}**")
    st.write(f"With the accuracy of **{confidence:.4f}**")