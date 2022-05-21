import torch
import torch.optim as optim
from dataloaders import get_mnist_dataloaders, get_paths, split_paths_between_train_and_val
from models import Generator, Discriminator
from training import Trainer

# Get data (1. all paths, 2. split train and validation, 3. dataloaders)
paths = get_paths(root_dir_path="C:\\Users\\jiri.kutalek\\Downloads\\og_2000-20220521T101142Z-001")
train_paths, val_paths = split_paths_between_train_and_val(paths, ratio=0.8)

# ------------------------------------------------------------------------------------------------

data_loader, _ = get_mnist_dataloaders(batch_size=64)
img_size = (32, 32, 1)

# Define generator and discriminator
generator = Generator(img_size=img_size, latent_dim=100, dim=16)
discriminator = Discriminator(img_size=img_size, dim=16)
print(generator)
print(discriminator)

# Initialize GD optimizers
lr = 1e-4
betas = (.9, .99)
G_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=betas)
D_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=betas)

# Train model
epochs = 200
trainer = Trainer(generator, discriminator, G_optimizer, D_optimizer, use_cuda=torch.cuda.is_available())
trainer.train(data_loader, epochs, save_training_gif=True)

# Save model
name = 'mnist_model'
torch.save(trainer.G.state_dict(), './gen_' + name + '.pt')
torch.save(trainer.D.state_dict(), './dis_' + name + '.pt')
