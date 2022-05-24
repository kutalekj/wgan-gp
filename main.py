#!/usr/bin/env python3
import argparse
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


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_parser():
    """
    Defined program parameters
    :return: Program parser
    """
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--loadModel', required=False, type=str2bool, dest='load_model', default=False)
    parser.add_argument('--saveModel', required=False, type=str2bool, dest='save_model', default=True)
    parser.add_argument('--load', required=False, type=str, dest='load', default="")
    parser.add_argument('--save', required=False, type=str, dest='save_name', default="")
    parser.add_argument('--epochs', dest='epochs', type=int, default=10)  # can be 'train' or 'test'
    return parser


if __name__ == '__main__':
    # Parse ARGS
    args = get_parser().parse_args()

    name = "cars_model"
    model_save_name = f'trainting_{name}.pt'

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

    if args.load_model:
        # Load from save
        if args.load == "":
            checkpoint = torch.load(model_save_name)
        else:
            checkpoint = torch.load(args.load)
        generator.load_state_dict(checkpoint['generator'])
        discriminator.load_state_dict(checkpoint['discriminator'])
        # Set to train mode
        generator.train()
        discriminator.train()

    # Initialize GD optimizers
    lr = 1e-4
    betas = (0.9, 0.99)
    G_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=betas)
    D_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=betas)

    # Train model
    epochs = args.epochs
    trainer = Trainer(generator, discriminator, G_optimizer, D_optimizer, use_cuda=torch.cuda.is_available())
    trainer.train(og_dataloader, grayscale_dataloader, epochs)

    if args.save_model:
        # Save model
        save_path = model_save_name
        if args.save_name != "":
            save_path = args.save_name
        torch.save({
            'generator': trainer.G.state_dict(),
            'discriminator': trainer.D.state_dict(),
        }, save_path)
