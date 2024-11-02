import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import os
import kagglehub
from collections import Counter

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

# Calculate class weights properly
labels = [label for _, label in dataset.samples]
class_counts = Counter(labels)
num_samples = len(labels)
class_weights = torch.FloatTensor([num_samples / (len(class_counts) * count) 
                                 for count in [class_counts[i] for i in range(len(class_counts))]])

# Create weighted sampler for training data
sample_weights = [class_weights[label] for label in labels]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

# Split dataset with fixed random seed
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
generator = torch.Generator().manual_seed(42)
train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)

# Create DataLoaders - only use sampler for training
train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
val_loader = DataLoader(val_ds, batch_size=batch_size)

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
        loss = F.cross_entropy(out, labels, weight=class_weights)
        return loss
    
    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels, weight=class_weights)
        acc = accuracy(out, labels)
        return {'val_loss': loss.detach(), 'val_acc': acc}
    
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
   def epoch_end(self, epoch, result):
    print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

# Move model and weights to device if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FabricDefectModel().to(device)
class_weights = class_weights.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def evaluate(model, val_loader):
    model.eval()
    outputs = []
    with torch.no_grad():
        for batch in val_loader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            outputs.append(model.validation_step((images, labels)))
    return model.validation_epoch_end(outputs)

def fit(epochs, model, train_loader, val_loader, optimizer):
    history = []
    for epoch in range(epochs):
        # Training Phase
        model.train()
        for batch in train_loader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            loss = model.training_step((images, labels))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        # Validation Phase
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result)
        history.append(result)
    return history

# Print dataset statistics before training
print("\nDataset Statistics:")
for i, (class_name, class_idx) in enumerate(dataset.class_to_idx.items()):
    count = class_counts[class_idx]
    weight = class_weights[class_idx].item()
    print(f"Class: {class_name}, Count: {count}, Weight: {weight:.4f}")

print("\nStarting training...")
history = fit(epochs, model, train_loader, val_loader, optimizer)

# Save model
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'history': history,
    'class_weights': class_weights,
    'class_to_idx': dataset.class_to_idx
}, 'fabric_defect_model.pth')

print("\nTraining completed!")
