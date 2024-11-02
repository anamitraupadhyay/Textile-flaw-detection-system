import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler, SubsetRandomSampler
import os
import kagglehub

# Download dataset
path = kagglehub.dataset_download("angelolmg/tilda-400-64x64-patches")
print("Path to dataset files:", path)
dataset_path = path

# Hyperparameters
batch_size = 64
epochs = 10
learning_rate = 0.001

# Transformations
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load dataset
dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

# Class weights for handling class imbalance
class_counts = torch.tensor([len(dataset.samples) for _, label in dataset.class_to_idx.items()])
class_weights = 1.0 / class_counts
sample_weights = [class_weights[target] for _, target in dataset.samples]

# Split dataset (using SubsetRandomSampler for better control)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_indices = list(range(train_size))
val_indices = list(range(train_size, len(dataset)))

train_sampler = WeightedRandomSampler(sample_weights[:train_size], len(train_indices), replacement=True)
val_sampler = SubsetRandomSampler(val_indices)


# Create DataLoaders
train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

# Model
class FabricDefectModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64 * 64, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 5)

    def forward(self, x):
        x = x.view(-1, 64 * 64)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

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


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

# Instantiate model, optimizer, and loss function
model = FabricDefectModel()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Evaluation function
def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

# Training loop
def fit(epochs, model, train_loader, val_loader, optimizer, criterion):
    history = []
    for epoch in range(epochs):
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result)
        history.append(result)
    return history

# Training
history = fit(epochs, model, train_loader, val_loader, optimizer, criterion)

# Save model
torch.save(model.state_dict(), 'fabric_defect_model.pth')
