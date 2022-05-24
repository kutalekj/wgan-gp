import numpy as np
import os
import torch
from random import shuffle
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image
from skimage.color import rgb2lab

# SIZE = 256


class CarDataset(Dataset):
    def __init__(self, paths, size=256):
        self.paths = paths
        self.size = size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = np.array(img)
        img_lab = rgb2lab(img).astype("float32")
        img_lab = transforms.ToTensor()(img_lab)

        # Normalizace do prostoru <-1,1>
        _L = img_lab[[0], ...] / 50.0 - 1.0  # Between -1 and 1
        ab = img_lab[[1, 2], ...] / 110.0  # Between -1 and 1

        return torch.cat([_L, ab], 0)
        # return {"L": _L, "ab": ab}


# class ColorizationDataset(Dataset):
#     def __init__(self, paths, split="train"):
#         if split == "train":
#             self.transforms = transforms.Compose(
#                 [
#                     transforms.Resize((SIZE, SIZE), Image.BICUBIC),
#                     transforms.RandomHorizontalFlip(),  # A little data augmentation!
#                 ]
#             )
#         elif split == "val":
#             self.transforms = transforms.Resize((SIZE, SIZE), Image.BICUBIC)

#         self.split = split
#         self.size = SIZE
#         self.paths = paths

#     # Read RGB image, convert to L*a*b, separate L channel from *a*b channels
#     def __getitem__(self, idx):
#         img = Image.open(self.paths[idx]).convert("RGB")
#         img = self.transforms(img)
#         img = np.array(img)
#         img_lab = rgb2lab(img).astype("float32")
#         img_lab = transforms.ToTensor()(img_lab)
#         _L = img_lab[[0], ...] / 50.0 - 1.0  # Between -1 and 1
#         ab = img_lab[[1, 2], ...] / 110.0  # Between -1 and 1

#         return {"L": _L, "ab": ab}

#     def __len__(self):
#         return len(self.paths)


# def get_cars_dataloader(batch_size=16, n_workers=0, pin_memory=True, **kwargs):
#     """Cars dataloader with (256, 256) sized images."""
#     dataset = ColorizationDataset(**kwargs)
#     dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers, pin_memory=pin_memory)
#     return dataloader


def get_paths(root_dir_path):
    paths = []
    # Get all file paths (recursive search in a directory)
    for path, currentDirectory, files in os.walk(root_dir_path):
        for file in files:
            paths.append(os.path.join(path, file))
    return paths


# def split_paths_between_train_and_val(paths, ratio):
#     # Randomly shuffle all data
#     shuffle(paths)

#     # Split paths by the ratio (e.g. for '0.8' split train:val by 80:20)
#     split_idx = int(len(paths) * ratio)
#     return paths[:split_idx], paths[split_idx:]


# def get_mnist_dataloaders(batch_size=128):
#     """MNIST dataloader with (32, 32) sized images."""
#     # Resize images so they are a power of 2
#     all_transforms = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
#     # Get train and test data
#     train_data = datasets.MNIST("../data", train=True, download=True, transform=all_transforms)
#     test_data = datasets.MNIST("../data", train=False, transform=all_transforms)
#     # Create dataloaders
#     train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
#     test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
#     return train_loader, test_loader
