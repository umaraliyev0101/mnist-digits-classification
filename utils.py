# utils.py
import matplotlib.pyplot as plt

def plot_image(image, label):
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title(f"Label: {label}")
    plt.axis('off')
    plt.show()