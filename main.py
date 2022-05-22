import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from PIL import Image
from dataloaders import get_mnist_dataloaders, get_paths, split_paths_between_train_and_val, get_cars_dataloader
from models import Generator, Discriminator
from training import Trainer

# Get data (1. all paths, 2. split train and validation, 3. dataloaders)
paths = get_paths(root_dir_path="C:\\Users\\jiri.kutalek\\Downloads\\og_2000-20220521T101142Z-001")
train_paths, val_paths = split_paths_between_train_and_val(paths, ratio=0.8)

train_dataloader = get_cars_dataloader(paths=train_paths, split='train')
validation_dataloader = get_cars_dataloader(paths=val_paths, split='val')

# data_loader, _ = get_mnist_dataloaders(batch_size=64)
# img_size = (32, 32, 1)

# ------------------------------------------------------------------------------------------------
# Debug plots/STDOUT prints
plt, axes = plt.subplots(4, 4, figsize=(10, 10))
for ax, img_path in zip(axes.flatten(), train_paths):
    ax.imshow(Image.open(img_path))
    ax.axis("off")
plt.show()

data = next(iter(train_dataloader))
Ls, abs_ = data['L'], data['ab']
img_size_L = (Ls.shape[2], Ls.shape[3], 1)
img_size_ab = (abs_.shape[2], abs_.shape[3], 2)
img_size = (img_size_L[0], img_size_L[1], 3)  # full 3-channel color image (L*a*b)
print(Ls.shape, abs_.shape)
print(len(train_dataloader), len(validation_dataloader))  # len(dataloader) = num_of_data / batch_size

# ------------------------------------------------------------------------------------------------
# Define generator and discriminator

# generator = Generator(img_size=img_size, latent_dim=100, dim=16)
# discriminator = Discriminator(img_size=img_size, dim=16)

generator = Generator(img_size=img_size_L, latent_dim=100, dim=16)
# discriminator = Discriminator(img_size=img_size, dim=16) TODO: JKU: Discriminator to take 3-chn. img on input (L*a*b)
discriminator = Discriminator(img_size=img_size_L, dim=16)
print(generator)
print(discriminator)

# ------------------------------------------------------------------------------------------------
# Initialize GD optimizers
lr = 1e-4
betas = (.9, .99)
G_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=betas)
D_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=betas)

# ------------------------------------------------------------------------------------------------
# Train model
epochs = 20
trainer = Trainer(generator, discriminator, G_optimizer, D_optimizer, use_cuda=torch.cuda.is_available())
trainer.train(train_dataloader, epochs, save_training_gif=True)
# trainer.train(data_loader, epochs, save_training_gif=False)

# ------------------------------------------------------------------------------------------------
# Save model
# name = 'mnist_model'
name = 'cars_model'
torch.save(trainer.G.state_dict(), './gen_' + name + '.pt')
torch.save(trainer.D.state_dict(), './dis_' + name + '.pt')
