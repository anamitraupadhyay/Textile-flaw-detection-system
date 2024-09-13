import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Define transformations
transform = transforms.Compose([
    transforms.Resize((28, 28)),  # Resize images to 28x28
    transforms.ToTensor(),        # Convert images to PyTorch tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize images
])

# Load the dataset
train_dataset = datasets.ImageFolder(root='data/train', transform=transform)
test_dataset = datasets.ImageFolder(root='data/test', transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Display some images from the dataset
def show_images(loader):
    data_iter = iter(loader)
    images, labels = data_iter.next()
    fig, axes = plt.subplots(1, 6, figsize=(12, 2))
    for i in range(6):
        ax = axes[i]
        img = images[i].numpy().transpose((1, 2, 0))
        img = (img * 0.5) + 0.5  # Unnormalize
        ax.imshow(img)
        ax.axis('off')
    plt.show()

show_images(train_loader)

# Now you can use train_loader and test_loader in your training and evaluation loops