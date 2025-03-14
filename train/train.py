# This script trains the DeepCNN model found in models.py


import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from dataset.dataloader import get_data_loaders
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pandas as pd
import pickle as pkl
from models import DeepCNN
print("imports complete")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


train_loader, test_loader, val_loader = get_data_loaders(batch_size=32)

train_images, train_labels = next(iter(train_loader))
val_images, val_labels = next(iter(val_loader))
test_images, test_labels = next(iter(test_loader))

print("got the data:")
print(train_images.shape)

# hyperp from optuna
dropout = 0.2229420772555262
model = DeepCNN(dropout=dropout).to(device)
criterion = nn.CrossEntropyLoss()
lr = 0.0004985912122180675
weight_decay = 6.90106735355982e-05
step_size = 3
gamma = 0.8278210415192079
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
print(f"loaded {model} with hyperparameters:")
print(f"initial learning rate: {lr}")
print(f"lr step size {step_size}, gamma {gamma}")
print(f"dropout {dropout}")
print(f"weight decay {weight_decay}")


print("training...")
epochs = 30

loss_plot_train = []
loss_plot_test = []
accuracy_train = []
accuracy_val = []

for epoch in range(epochs):
    
    model.train()
    running_loss = 0.0
    correct_train, total_train = 0, 0
    
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        
        # compute training accuracy
        _, predicted = torch.max(outputs, 1) 
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)

    # compute average train loss and accuracy
    avg_train_loss = running_loss / len(train_loader.dataset)
    train_accuracy = correct_train / total_train
    loss_plot_train.append(avg_train_loss)
    accuracy_train.append(train_accuracy)

    scheduler.step()

    # validation step
    model.to(device)
    model.eval()
    running_val_loss = 0.0
    correct_val, total_val = 0, 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item() * images.size(0)  # Accumulate batch loss
            
            _, predicted = torch.max(outputs, 1)  # Get predicted class
            correct_val += (predicted == labels).sum().item()
            total_val += labels.size(0)
    
    avg_val_loss = running_val_loss / len(val_loader.dataset)
    val_accuracy = correct_val / total_val
    loss_plot_test.append(avg_val_loss)
    accuracy_val.append(val_accuracy)

    print(f"Epoch [{epoch+1}/{epochs}] -- "
          f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.4f} | "
          f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.4f} | "
          f"LR: {scheduler.get_last_lr()[0]:.6f}")
    
print(f"Final Training Accuracy: {accuracy_train[-1]:.4f}")
print(f"Final Validation Accuracy: {accuracy_val[-1]:.4f}")

torch.save(model.to(device).state_dict(), 'model_weights.pth')
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'train_loss': loss_plot_train,
    'val_loss': loss_plot_test
}, "checkpoint.pth")

print("model/checkpoint saved")

loss_df = pd.DataFrame({
    'epoch': list(range(1, len(loss_plot_train) + 1)),
    'train_loss': loss_plot_train,
    'train_accuracy': accuracy_train,
    'test_loss': loss_plot_test,
    'test_accuracy': accuracy_val
})
loss_df.to_csv('loss.csv', index=False)
print("losses saved")

