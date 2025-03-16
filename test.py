import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import MNIST_Classifier

# Load the trained model
model = MNIST_Classifier()
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
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")