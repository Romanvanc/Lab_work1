import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_test_transform

def download_dataset():
    import kagglehub
    path = kagglehub.dataset_download("jessicali9530/stanford-dogs-dataset")
    return path

def load_datasets(data_path=None):
    if data_path is None:
        data_path = 'stanford_dogs/images/Images'
    
    train_transform, val_test_transform = get_transforms()
    
    dataset = datasets.ImageFolder(root=data_path, transform=train_transform)
    num_classes = len(dataset.classes)
    class_names = dataset.classes
    
    total = len(dataset)
    train_size = int(0.7 * total)
    val_size = int(0.15 * total)
    test_size = total - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(497328)
    )
    
    val_dataset.dataset.transform = val_test_transform
    test_dataset.dataset.transform = val_test_transform
    
    print(f"Total images: {total}")
    print(f"Train: {train_size}, Validation: {val_size}, Test: {test_size}")
    print(f"Number of classes: {num_classes}")
    
    return train_dataset, val_dataset, test_dataset, num_classes, class_names

def create_loaders(train_dataset, val_dataset, test_dataset, batch_size=32):
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
