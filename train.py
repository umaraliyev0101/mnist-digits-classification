import torch
from torchvision import datasets, transforms
import torch.nn as nn  # Add this line
import torch.optim as optim  # Add this line
from torch.utils.data import DataLoader 
from model import MNIST_Classifier

# Hyperparameters
batch_size = 64
learning_rate = 0.001
epochs = 10

# Load and preprocess data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_data = datasets.MNIST(
    root='data', 
    train=True, 
    download=True, 
    transform=transform
)

train_loader = DataLoader(
    train_data, 
    batch_size=batch_size, 
    shuffle=True)


# Initialize model, loss function, and optimizer
model = MNIST_Classifier()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), 'mnist_model.pth')