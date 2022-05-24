#!/usr/bin/env python3

import enum
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from PIL import Image

from configuration import config
from dataloaders import CarDataset, get_paths
from models import Generator, Discriminator, UnetGenerator
from training import Trainer

dataset_paths = config(section="dataset_paths")
training_settings = config(section="training_settings")


og_paths = get_paths(root_dir_path=dataset_paths["og"])
og_dataset = CarDataset(og_paths)
og_dataloader = DataLoader(og_dataset, batch_size=int(training_settings["batch_size"]), shuffle=True)


grayscale_paths = get_paths(root_dir_path=dataset_paths["grayscale"])
grayscale_dataset = CarDataset(grayscale_paths)
grayscale_dataloader = DataLoader(grayscale_dataset, batch_size=int(training_settings["batch_size"]), shuffle=True)


img_size = (256, 256, 3)

generator = UnetGenerator(input_c=3, output_c=2)
discriminator = Discriminator(img_size=img_size, dim=16)

# Initialize GD optimizers
lr = 1e-4
betas = (0.9, 0.99)
G_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=betas)
D_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=betas)


# Train model
epochs = 10
trainer = Trainer(generator, discriminator, G_optimizer, D_optimizer, use_cuda=torch.cuda.is_available())
trainer.train(og_dataloader, grayscale_dataloader, epochs)


# # Save model
# # name = 'mnist_model'
name = "cars_model"
torch.save(trainer.G.state_dict(), "./gen_" + name + ".pt")
torch.save(trainer.D.state_dict(), "./dis_" + name + ".pt")
