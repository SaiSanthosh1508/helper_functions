
import torch
import os
import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader

def create_dataloaders(train_dir: str,
                       test_dir: str,
                       transforms,
                       batch_size=32,
                       seed=42,
                       num_workers=os.cpu_count()):
    # Load train and test datasets using ImageFolder
    train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=transforms)
    test_dataset = torchvision.datasets.ImageFolder(test_dir, transform=transforms)
    
    # Create DataLoaders
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,  # Shuffle the training data
                                  num_workers=num_workers,
                                  pin_memory=True)
    
    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,  # Do not shuffle test data
                                 num_workers=num_workers,
                                 pin_memory=True)
    
    # Extract class names
    class_names = train_dataset.classes

    return train_dataloader, test_dataloader, class_names
