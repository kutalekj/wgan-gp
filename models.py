import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Generator(nn.Module):
    def __init__(self, img_size, latent_dim, dim):
        super(Generator, self).__init__()

        self.dim = dim
        self.latent_dim = latent_dim
        self.img_size = img_size
        # e.g. (2,2) if img_size = (32, 32, 1)
        self.feature_sizes = (self.img_size[0] / 16, self.img_size[1] / 16)

        # HINT: Linear(in_features, out_features)
        self.latent_to_features = nn.Sequential(
            nn.Linear(latent_dim, int(8 * dim * self.feature_sizes[0] * self.feature_sizes[1])),
            nn.ReLU()
        )

        # HINT: ConvTranspose2d(in_features, out_features, kernel_size, stride, padding, ...)
        # HINT: BatchNorm2d(num_features)
        self.features_to_image = nn.Sequential(
            nn.ConvTranspose2d(8 * dim, 4 * dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(4 * dim),
            nn.ConvTranspose2d(4 * dim, 2 * dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(2 * dim),
            nn.ConvTranspose2d(2 * dim, dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(dim),
            nn.ConvTranspose2d(dim, self.img_size[2], kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, input_data):
        # Map latent into appropriate size for transposed convolutions
        x = self.latent_to_features(input_data)
        # Reshape
        x = x.view(-1, 8 * self.dim, int(self.feature_sizes[0]), int(self.feature_sizes[1]))
        # Return generated image
        return self.features_to_image(x)

    def sample_latent(self, num_samples):
        # Returns a tensor filled with random numbers from a normal distribution (parameters define the tensor shape)
        return torch.randn((num_samples, self.latent_dim))


class Discriminator(nn.Module):
    def __init__(self, img_size, dim):
        """
        img_size : (int, int, int)
            Height and width must be powers of 2.  E.g. (32, 32, 1) or
            (64, 128, 3). Last number indicates number of channels, e.g. 1 for
            grayscale or 3 for RGB
        """
        super(Discriminator, self).__init__()

        self.img_size = img_size

        # HINT: LeakyReLU(negative_slope)
        self.image_to_features = nn.Sequential(
            nn.Conv2d(self.img_size[2], dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim, 2 * dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(2 * dim, 4 * dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(4 * dim, 8 * dim, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

        # 4 convolutions of stride 2 - half the size everytime -> output size = 8 * (img_size / 2^4) * (img_size / 2^4)
        output_size = int(8 * dim * (img_size[0] / 16) * (img_size[1] / 16))
        self.features_to_prob = nn.Sequential(
            nn.Linear(output_size, 1),
            nn.Sigmoid()
        )

    def forward(self, input_data):
        batch_size = input_data.size()[0]
        x = self.image_to_features(input_data)
        x = x.view(batch_size, -1)
        return self.features_to_prob(x)
