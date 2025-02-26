import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_data_loaders(base_path='/projects/dsci410-510/', batch_size=256):
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(root=os.path.join(base_path, 'train'), transform=transform)
    test_dataset = datasets.ImageFolder(root=os.path.join(base_path, 'test'), transform=transform)
    val_dataset = datasets.ImageFolder(root=os.path.join(base_path, 'val'), transform=transform)

    #print(f"Class mapping: {train_dataset.class_to_idx}")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    return train_loader, test_loader, val_loader

# Example usage
if __name__ == "__main__":
    train_loader, test_loader, val_loader = get_data_loaders()

    for images, labels in train_loader:
        print(f"train batch shape: {images.shape}")
        print(f"Labels: {labels}")
        break
