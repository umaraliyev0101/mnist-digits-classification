import torch
import torch.nn as nn
import torch.nn.functional as F

class MNIST_Classifier(nn.Module):
    """
    A Convolutional Neural Network (CNN) for classifying MNIST digits.

    Architecture:
    - 2 convolutional layers
    - 2 fully connected layers
    - Dropout for regularization
    """
    def __init__(self):
        super().__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 1 input channel (grayscale), 32 filters
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 7x7 image after pooling
        self.fc2 = nn.Linear(128, 10)  # 10 output classes (digits 0-9)
        
        # Dropout to prevent overfitting
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 10).
        """
        # Convolution + ReLU + MaxPool
        x = F.relu(self.conv1(x))  # Output shape: (32, 28, 28)
        x = F.max_pool2d(x, 2)      # Output shape: (32, 14, 14)
        
        x = F.relu(self.conv2(x))  # Output shape: (64, 14, 14)
        x = F.max_pool2d(x, 2)      # Output shape: (64, 7, 7)
        
        # Flatten for fully connected layers
        x = x.view(-1, 64 * 7 * 7)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x