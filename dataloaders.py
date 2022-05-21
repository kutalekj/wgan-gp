import numpy as np
import os
from random import shuffle
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


def get_paths(root_dir_path):
    paths = []
    # Get all file paths (recursive search in a directory)
    for path, currentDirectory, files in os.walk(root_dir_path):
        for file in files:
            paths.append(os.path.join(path, file))
    return paths


def split_paths_between_train_and_val(paths, ratio):
    # Randomly shuffle all data
    shuffle(paths)

    # Split paths by the ratio (e.g. for '0.8' split train:val by 80:20)
    split_idx = int(len(paths) * ratio)
    return paths[:split_idx], paths[split_idx:]


def get_mnist_dataloaders(batch_size=128):
    """MNIST dataloader with (32, 32) sized images."""
    # Resize images so they are a power of 2
    all_transforms = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
    # Get train and test data
    train_data = datasets.MNIST('../data', train=True, download=True, transform=all_transforms)
    test_data = datasets.MNIST('../data', train=False, transform=all_transforms)
    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


def get_fashion_mnist_dataloaders(batch_size=128):
    """Fashion MNIST dataloader with (32, 32) sized images."""
    # Resize images so they are a power of 2
    all_transforms = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
    # Get train and test data
    train_data = datasets.FashionMNIST('../fashion_data', train=True, download=True, transform=all_transforms)
    test_data = datasets.FashionMNIST('../fashion_data', train=False, transform=all_transforms)
    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


def get_lsun_dataloader(path_to_data='../lsun', dataset='bedroom_train', batch_size=64):
    """LSUN dataloader with (128, 128) sized images.

    path_to_data : str
        One of 'bedroom_val' or 'bedroom_train'
    """
    # Compose transforms
    transform = transforms.Compose([transforms.Resize(128), transforms.CenterCrop(128), transforms.ToTensor()])

    # Get dataset
    lsun_dataset = datasets.LSUN(db_path=path_to_data, classes=[dataset], transform=transform)

    # Create dataloader
    return DataLoader(lsun_dataset, batch_size=batch_size, shuffle=True)
