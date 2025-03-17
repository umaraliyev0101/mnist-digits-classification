import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import MNIST_Classifier
import matplotlib.pyplot as plt

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
model = MNIST_Classifier().to(device)
model.load_state_dict(torch.load('mnist_model.pth',weights_only=True))
model.eval()

# Load and preprocess test data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# Test the model
correct = 0
total = 0

all_labels = []
all_predictions = []

# Inside test.py  
import os  
os.makedirs("errors", exist_ok=True)  

with torch.no_grad():
    for idx, (images, labels) in enumerate(test_loader):
        # Move data to the correct device
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Store labels and predictions for evaluation
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

        # Save misclassified images
        for i in range(len(labels)):
            if predicted[i] != labels[i]:
                image_np = images[i].squeeze().cpu().numpy()  # Convert to numpy array
                plt.imsave(f"errors/{idx}_{i}_pred{predicted[i]}_true{labels[i]}.png", image_np, cmap='gray')

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")