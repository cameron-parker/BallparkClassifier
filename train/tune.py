import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models import DeepCNN
import os
from tqdm import tqdm


talapaspath = '/projects/dsci410_510/ballpark'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epochs = 30
n_trials = 10 

def objective(trial):
    # search space
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)  # Log scale for LR
    dropout = trial.suggest_float("dropout", 0.2, 0.5)  # Dropout range
    step_size = trial.suggest_int("step_size", 1, 10) # learning rate step size
    gamma = trial.suggest_float("gamma", 0.1, 0.9) # learning rate gamma
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)  # Regularization
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])  # Fixed choices

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((180, 320), scale=(0.95, 1.0)),  # Less aggressive cropping
        transforms.RandomHorizontalFlip(p=0.5),  # Keep horizontal flip
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),  # Small shifts
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Less extreme
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # test, val transforms w/ no augmentation
    test_val_transform = transforms.Compose([
        transforms.Resize((180, 320)),  # Keep fixed size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print(f"DeepCNN TRIAL [{trial.number}]: grabbing data and creating dataloaders")
    train_dataset = datasets.ImageFolder(root=os.path.join(talapaspath, 'train'), transform=train_transform)
    val_dataset = datasets.ImageFolder(root=os.path.join(talapaspath, 'val'), transform=test_val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)

    # initialize model
    model = DeepCNN(dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    print(f"DeepCNN TRIAL [{trial.number}]: training...")
    # train
    for epoch in range(num_epochs):
        model.train()
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        scheduler.step()

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        print(f"DeepCNN TRIAL [{trial.number}] epoch [{epoch+1}/{num_epochs}] complete")
    return val_loss / len(val_loader)  # Return average validation loss



study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=4))
study.optimize(objective, n_trials=n_trials, n_jobs=1)

print("Best hyperparameters for DeepCNN:", study.best_params)