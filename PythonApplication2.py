import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os

# Hyperparameters
batch_size = 64
epochs = 10
learning_rate = 0.001

# Path to your dataset folder (adjust this to your dataset location)
dataset_path = 'path_to_your_dataset_folder'  # Replace this with the correct path

# Transformations: Convert images to grayscale, resize if necessary, and normalize
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Grayscale conversion
    transforms.ToTensor(),  # Convert images to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize with mean and std deviation
])

# Load dataset using ImageFolder (Make sure your dataset is organized in folders)
dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

# Split dataset into training (80%) and validation (20%) sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

# Create DataLoaders for training and validation
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size)

# Define the model
class FabricDefectModel(nn.Module):
    def __init__(self):
        super(FabricDefectModel, self).__init__()
        self.linear1 = nn.Linear(64*64, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 5)  # 5 output classes ('good', 'hole', etc.)

    def forward(self, xb):
        xb = xb.view(-1, 64*64)  # Flatten the 64x64 image
        xb = F.relu(self.linear1(xb))
        xb = F.relu(self.linear2(xb))
        out = self.linear3(xb)
        return out

    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print(f"Epoch [{epoch}], val_loss: {result['val_loss']:.4f}, val_acc: {result['val_acc']:.4f}")

# Define accuracy function
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

# Instantiate the model
model = FabricDefectModel()

# Define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Evaluation function
def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

# Training loop
def fit(epochs, model, train_loader, val_loader, optimizer):
    history = []

    for epoch in range(epochs):
        # Training Phase
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Validation Phase
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result)
        history.append(result)

    return history

# Start training the model
history = fit(epochs, model, train_loader, val_loader, optimizer)

# Save the model after training
torch.save(model.state_dict(), 'fabric_defect_model.pth')
