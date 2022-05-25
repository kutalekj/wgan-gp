"""User guided car image colorization with WGAN-GP.

Authors
-------
Chimenti Andrea     xchime00
Chudarek Ales       xchuda04
Kosak Vaclav        xkosak01
Kutalek Jiri        xkutal09

Insitution
----------
Brno University of Technology
Faculty of Information Technology

Date
----
May 2022

"""

import numpy as np
import os
import torch
from random import shuffle
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image
from skimage.color import rgb2lab


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


def get_paths(root_dir_path):
    paths = []
    # Get all file paths (recursive search in a directory)
    for path, currentDirectory, files in os.walk(root_dir_path):
        for file in files:
            paths.append(os.path.join(path, file))
    return paths
