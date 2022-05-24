#!/usr/bin/env python3

import enum
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from PIL import Image

from configuration import config
from dataloaders import CarDataset, get_paths
from models import Generator, Discriminator
from training import Trainer

dataset_paths = config(section="dataset_paths")
training_settings = config(section="training_settings")


og_paths = get_paths(root_dir_path=dataset_paths["og"])
og_dataset = CarDataset(og_paths)
og_dataloader = DataLoader(og_dataset, batch_size=int(training_settings["batch_size"]), shuffle=True)


# d = next(iter(og_dataloader))

# print(d["ab"].size())

d = next(enumerate(og_dataloader))

print(d[1]["L"].size())
