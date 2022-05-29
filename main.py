#!/usr/bin/env python3

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

import argparse
import enum
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from PIL import Image

import models
from configuration import config
from dataloaders import CarDataset, get_paths
from models import Discriminator, UnetGenerator
from training import Trainer


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_parser():
    """
    Defined program parameters
    :return: Program parser
    """
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--loadModel", required=False, type=str2bool, dest="load_model", default=False)
    parser.add_argument("--saveModel", required=False, type=str2bool, dest="save_model", default=True)
    parser.add_argument("--load", required=False, type=str, dest="load", default="")
    parser.add_argument("--save", required=False, type=str, dest="save_name", default="")
    parser.add_argument("--epochs", dest="epochs", type=int, default=10)  # can be 'train' or 'test'
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=16)
    return parser


if __name__ == "__main__":
    # Parse ARGS
    args = get_parser().parse_args()

    name = "cars_model"
    model_save_name = f"training_{name}.pt"

    dataset_paths = config(section="dataset_paths")

    og_paths = get_paths(root_dir_path=dataset_paths["og"])
    og_dataset = CarDataset(og_paths)
    og_dataloader = DataLoader(og_dataset, batch_size=args.batch_size, shuffle=True)

    grayscale_paths = get_paths(root_dir_path=dataset_paths["grayscale"])
    grayscale_paths = grayscale_paths[:2000]
    grayscale_dataset = CarDataset(grayscale_paths)
    grayscale_dataloader = DataLoader(grayscale_dataset, batch_size=args.batch_size, shuffle=True)

    img_size = (256, 256, 3)

    generator = UnetGenerator(input_c=3, output_c=2)
    discriminator = Discriminator(img_size=img_size, dim=16)

    generator = models.init_weights(generator)
    discriminator = models.init_weights(discriminator)

    print(generator)
    print(discriminator)

    if args.load_model:
        # Load from save
        if args.load == "":
            checkpoint = torch.load(model_save_name)
        else:
            checkpoint = torch.load(args.load)
        generator.load_state_dict(checkpoint["generator"])
        discriminator.load_state_dict(checkpoint["discriminator"])
        # Set to train mode
        generator.train()
        discriminator.train()

    # Initialize GD optimizers
    lr = 1e-4
    betas = (0.9, 0.99)
    G_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=betas)
    D_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=betas)

    save_path = ""
    if args.save_model:
        save_path = model_save_name
        if args.save_name != "":
            save_path = args.save_name

    # Train model
    epochs = args.epochs
    trainer = Trainer(generator, discriminator, G_optimizer, D_optimizer, use_cuda=torch.cuda.is_available())
    trainer.train(og_dataloader, grayscale_dataloader, epochs, save_path=save_path)

    # Save model
    if args.save_model:
        torch.save(
            {
                "generator": trainer.G.state_dict(),
                "discriminator": trainer.D.state_dict(),
            },
            save_path,
        )
