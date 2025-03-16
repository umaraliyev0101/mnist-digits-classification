import torch
from torchvision import transforms
from PIL import Image
from model import MNIST_Classifier

# Load the trained model
model = MNIST_Classifier()
model.load_state_dict(torch.load('models/mnist_model.pth', weights_only=True))
model.eval()

# Preprocess the input image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = Image.open(image_path)
    return transform(image).unsqueeze(0)

# Predict the digit
def predict(image_path):
    image = preprocess_image(image_path)
    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)
        confidence, prediction = torch.max(probabilities, dim=1)
        confidence = confidence.item()  # Convert to Python float
        prediction = prediction.item()  # Convert to Python int
        result = [confidence, prediction]
    return result

# Example usage
if __name__ == "__main__":
    image_path = "images/3.jpg"
    result = predict(image_path)
    print(f"Predicted Digit: {result[1]}")
    print(f"With the accuracy of {result[0]}")