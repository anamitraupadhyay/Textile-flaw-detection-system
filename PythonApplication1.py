import torch
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
from torch.utils.data import random_split, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

print(torch.__version__)
# Define the accuracy function
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

# Download MNIST dataset
dataset = MNIST(root='data/', download=True)
print(len(dataset))

# Create a test dataset
test_dataset = MNIST(root='data/', download=False, train=False)
print(len(test_dataset))

# Display an image from the dataset
image, label = dataset[59999]
plt.imshow(image, cmap='gray')
print("label", label)

# Transform dataset to tensor
transform = transforms.ToTensor()
dataset = MNIST(root='data/', download=True, transform=transform)
img_tensor, label = dataset[59999]
print(img_tensor.shape, label)

# Split dataset into training and validation sets
train_ds, val_ds = random_split(dataset, [50000, 10000])
print(len(train_ds))
print(len(val_ds))
print(img_tensor.shape, label)

# Taking a batch size for the validation set using DataLoader for splitting the dataset
batch_size = 128
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size)

# Declaring the size of the images and the no. of digits the number might fall into i.e. 0 to 9 is numclasses
ip_size = 28 * 28
num_classes = 10

# Creating a logistic regression model using nn.Linear
model = nn.Linear(ip_size, num_classes)

# Print the shapes of weight and bias
print(model.weight.shape)
print(model.bias.shape)

# Forward pass through the model
for images, labels in train_loader:
    print(labels)
    print(images.shape)
    outputs = model(images.view(-1, 28 * 28))
    print(outputs)
    break

# Define a custom model
class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()
        self.linear = nn.Linear(ip_size, num_classes).float()  # Set the data type here

    def forward(self, xb):
        xb = xb.view(-1, 28 * 28)
        out = self.linear(xb)
        return out

    def training_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return {'val_loss': loss, 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))

model = MnistModel()
print(model.linear)
for images, labels in train_loader:
    outputs = model(images)
    break
print("op.shapes", outputs.shape)
print("sample op", outputs[:2].data)
print(images.size())
probs = F.softmax(outputs, dim=1)
print(probs[:2].data)
print(torch.sum(probs[0]).item())
max_probs, preds = torch.max(probs, dim=1)
print(preds)
print(max_probs)
print(labels)

print(preds == labels)
print(torch.sum(preds == labels))
print(accuracy(outputs, labels))
Loss_fn = F.cross_entropy(outputs, labels)

print(Loss_fn)

# Training the model
def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    optimizer = opt_func(model.parameters(), lr)
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

# Upto history
result0 = evaluate(model, val_loader)
print(result0)

history1 = fit(5, 0.001, model, train_loader, val_loader)
history2 = fit(5, 0.001, model, train_loader, val_loader)
history3 = fit(5, 0.001, model, train_loader, val_loader)
history4 = fit(5, 0.001, model, train_loader, val_loader)
history5 = fit(5, 0.001, model, train_loader, val_loader)
history6 = fit(5, 0.001, model, train_loader, val_loader)

history = [result0] + history1 + history2 + history3 + history4 + history5 + history6
accuracies = [result['val_acc'] for result in history]
plt.plot(accuracies, '-x')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('accuracy vs no. of epochs')
plt.show()

# Define the custom_collate function
def custom_collate(batch):
    # Convert PIL Images to tensors
    images, labels = zip(*batch)
    images = [transforms.ToTensor()(img) for img in images]
    return torch.stack(images), torch.tensor(labels)

# Define the predict_image function
def predict_image(img, model):
    # Convert the NumPy array to a PyTorch tensor and ensure it has the correct dtype
    xb = torch.from_numpy(img).float().unsqueeze(0)
    yb = model(xb)
    _, preds = torch.max(yb, dim=1)
    return preds[0].item()

# Display an image from the dataset and make a prediction
img, label = test_dataset[0]

# Convert the 'Image' object to a NumPy array
img_array = np.array(img)

plt.imshow(img_array, cmap='gray')  # Displaying the NumPy array directly
print("shape:", img_array.shape, ', predicted:', predict_image(img_array, model))
plt.show()

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
result = evaluate(model, test_loader)
print(result)

# Saving and loading the model
torch.save(model.state_dict(), 'mnist.pth')
model2 = MnistModel()

# Comparing between randomly initialized weights, biases, and saved trained weights, biases
print("Evaluation before loading the state:")
print(evaluate(model2, test_loader))

# Load the saved state
model2.load_state_dict(torch.load('mnist.pth'))

# Train the model2 (optional if you want to fine-tune it)
# fit(5, 0.001, model2, train_loader, val_loader)

print("Evaluation after loading the state:")
result = evaluate(model2, test_loader)
print(result)  # Difference between 2 prints

# Fine-tuning the loaded model (optional)
fine_tune_epochs = 5
fine_tune_lr = 0.0001
fine_tune_optimizer = torch.optim.Adam

fine_tune_history = fit(fine_tune_epochs, fine_tune_lr, model2, train_loader, val_loader, opt_func=fine_tune_optimizer)

# Print fine-tuning results
print("Fine-tuning results:")
print(fine_tune_history)

# Evaluate the model after fine-tuning
print("Evaluation after fine-tuning:")
result_after_fine_tune = evaluate(model2, test_loader)
print(result_after_fine_tune)
