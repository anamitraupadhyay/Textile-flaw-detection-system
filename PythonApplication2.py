import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler, SubsetRandomSampler
import kagglehub
from collections import Counter
from PIL import Image

# --- Data Loading and Preprocessing ---
#Download dataset (only run this once)
path = kagglehub.dataset_download("angelolmg/tilda-400-64x64-patches")
print("Path to dataset files:", path)
dataset_path = path

# For testing purposes, replace with your local path
dataset_path = "/home/gitpod/.cache/kagglehub/datasets/angelolmg/tilda-400-64x64-patches/versions/1" #Replace with your path

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

# Calculate class weights
labels = [label for _, label in dataset.samples]
class_counts = Counter(labels)
num_samples = len(labels)
class_weights = torch.FloatTensor([num_samples / (len(class_counts) * count)
                                 for count in [class_counts[i] for i in range(len(class_counts))]])
sample_weights = [class_weights[label] for label in labels]

# Split dataset indices (80% train, 10% validation, 10% test)
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_indices = list(range(train_size))
val_indices = list(range(train_size, train_size + val_size))
test_indices = list(range(train_size + val_size, len(dataset)))

# Create samplers
train_sampler = WeightedRandomSampler(sample_weights[:train_size], len(train_indices), replacement=True)
val_sampler = SubsetRandomSampler(val_indices)
test_sampler = SubsetRandomSampler(test_indices)

# Create DataLoaders
train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

# --- Model Definition ---
class FabricDefectModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64 * 64, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 5)  # Assuming 5 classes

    def forward(self, x):
        x = x.view(-1, 64 * 64)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels, weight=class_weights.to(images.device))
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels, weight=class_weights.to(images.device))
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

# --- Training and Evaluation ---
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
        model.train()
        for batch in train_loader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            loss = model.training_step((images, labels))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result)
        history.append(result)
    return history

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels, weight=class_weights)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss /= len(test_loader)
    test_accuracy = 100 * correct / total
    print(f"\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")


# --- Patch Processing and Prediction ---
def process_patches(model, image_paths, class_to_idx):
    patch_predictions = []
    for image_path in image_paths:
        try:
            patch = Image.open(image_path).convert('L')
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            patch_tensor = transform(patch).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(patch_tensor)
                _, predicted_index = torch.max(output, 1)
                predicted_label = list(class_to_idx.keys())[list(class_to_idx.values()).index(predicted_index.item())]
                patch_predictions.append(predicted_label)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    prediction_counts = Counter(patch_predictions)
    most_common_prediction = prediction_counts.most_common(1)[0][0] if prediction_counts else None
    return most_common_prediction


def predict_from_directory(model, directory, class_to_idx, num_patches=64):
    image_paths = [os.path.join(directory, f) for f in os.listdir(directory)][:num_patches]
    prediction = process_patches(model, image_paths, class_to_idx)
    return prediction


# --- Main Execution Block ---
if __name__ == "__main__":
    # Print dataset statistics
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

    # Test the model
    checkpoint = torch.load('fabric_defect_model.pth')
    model = FabricDefectModel().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    class_weights = checkpoint['class_weights'].to(device)
    test(model, test_loader)

    # Get prediction from a directory
    test_dir = input("Enter the path to the directory containing patches for prediction: ")
    prediction = predict_from_directory(model, test_dir, checkpoint['class_to_idx'])

    if prediction:
        print(f"\nPrediction for the directory: {prediction}")
    else:
        print("\nNo prediction could be made.")

    print("\nAll processes completed!")
