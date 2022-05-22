import imageio
import numpy as np
import torch
from torchvision.utils import make_grid
from torch.autograd import Variable
from torch.autograd import grad as torch_grad


class Trainer:
    def __init__(self, generator, discriminator, gen_optimizer, dis_optimizer,
                 gp_weight=10, critic_iterations=5, print_every=5, use_cuda=False):
        self.G = generator
        self.G_opt = gen_optimizer
        self.D = discriminator
        self.D_opt = dis_optimizer
        self.losses = {'G': [], 'D': [], 'GP': [], 'gradient_norm': []}
        self.num_steps = 0
        self.use_cuda = use_cuda
        self.gp_weight = gp_weight
        self.critic_iterations = critic_iterations
        self.print_every = print_every

        if self.use_cuda:
            self.G.cuda()
            self.D.cuda()

    def _critic_train_iteration(self, data):
        # Get generated data
        batch_size = data.size()[0]
        generated_data = self.sample_generator(batch_size)

        # Calculate probabilities on real and generated data
        # TODO: JKU: Modify generator to concatenate input (L*a*b) with noise and output a 2-channel (*a*b) image
        data = Variable(data)
        if self.use_cuda:
            data = data.cuda()
        d_real = self.D(data)
        d_generated = self.D(generated_data)

        # Calculate the gradient penalty
        gradient_penalty = self._gradient_penalty(data, generated_data)
        self.losses['GP'].append(gradient_penalty.data)

        # Get total loss and optimize
        # https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
        self.D_opt.zero_grad()
        d_loss = d_generated.mean() - d_real.mean() + gradient_penalty
        d_loss.backward()

        # Perform a single optimization step
        self.D_opt.step()
        self.losses['D'].append(d_loss.data)

    def _generator_train_iteration(self, data):
        # https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
        self.G_opt.zero_grad()

        # Get generated data
        batch_size = data.size()[0]
        generated_data = self.sample_generator(batch_size)

        # Calculate loss and optimize
        d_generated = self.D(generated_data)
        g_loss = - d_generated.mean()
        g_loss.backward()

        # Perform a single optimization step
        self.G_opt.step()
        self.losses['G'].append(g_loss.data)

    def _gradient_penalty(self, real_data, generated_data):
        batch_size = real_data.size()[0]

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1, 1, 1)
        alpha = alpha.expand_as(real_data)
        if self.use_cuda:
            alpha = alpha.cuda()
        interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
        interpolated = Variable(interpolated, requires_grad=True)
        if self.use_cuda:
            interpolated = interpolated.cuda()

        # Calculate probability of interpolated examples
        prob_interpolated = self.D(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(
                                   prob_interpolated.size()).cuda() if self.use_cuda else torch.ones(
                                   prob_interpolated.size()),
                               create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # So, flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)
        self.losses['gradient_norm'].append(gradients.norm(2, dim=1).mean().data)

        # Derivatives of the gradient close to 0 can cause problems because of the square root
        # So, manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return self.gp_weight * ((gradients_norm - 1) ** 2).mean()

    def _train_epoch(self, data_loader):
        # A single training epoch
        for i, data in enumerate(data_loader):
            self.num_steps += 1
            # Discriminator training?
            self._critic_train_iteration(data['L'])  # TODO: JKU: A 3-channel REAL image (L*a*b) should be passed in
            # self._critic_train_iteration(data[0])

            # Only update generator every |critic_iterations| iterations (every second iteration?)
            if self.num_steps % self.critic_iterations == 0:
                # Generator training?
                self._generator_train_iteration(data['L'])  # TODO: JKU: A 3-chn. USER image (L*a*b) should be passed in
                # self._generator_train_iteration(data[0])
                # TODO: JKU: Concatenate 2-chn. (*a,*b) generator output with the 1-chn. grayscale prior ('L' from REAL)

            # STDOUT print
            if i % self.print_every == 0:
                print("Iteration {}".format(i + 1))
                print("D: {}".format(self.losses['D'][-1]))
                print("GP: {}".format(self.losses['GP'][-1]))
                print("Gradient norm: {}".format(self.losses['gradient_norm'][-1]))
                if self.num_steps > self.critic_iterations:
                    print("G: {}".format(self.losses['G'][-1]))

    def train(self, data_loader, epochs, save_training_gif=True):
        fixed_latents = []
        training_progress_images = []

        # Visualization of the training progress
        if save_training_gif:
            # Fix latents to see how image generation improves during training
            fixed_latents = Variable(self.G.sample_latent(64))  # TODO: JKU: Why hardcoded 64? Is it batch_size?
            if self.use_cuda:
                fixed_latents = fixed_latents.cuda()

        # Main training loop
        for epoch in range(epochs):
            print("\nEpoch {}".format(epoch + 1))
            self._train_epoch(data_loader)

            # Visualization of the training progress
            if save_training_gif:
                # Generate batch of images and convert to grid
                img_grid = make_grid(self.G(fixed_latents).cpu().data)
                # Convert to numpy and transpose axes to fit imageio convention (e.g. (width, height, channels))
                img_grid = np.transpose(img_grid.numpy(), (1, 2, 0))
                # Add image grid to training progress
                training_progress_images.append(img_grid)

        # Visualization of the training progress
        if save_training_gif:  # TODO: JKU: Make smaller image grid/lower batch_size/... (too many samples in final GIF)
            imageio.mimsave('./training_{}_epochs.gif'.format(epochs), training_progress_images)

    def sample_generator(self, num_samples):
        # "Variable" is a wrapper around a PyTorch Tensor, representing a node in a computational graph
        latent_samples = Variable(self.G.sample_latent(num_samples))  # Get gaussian noise data
        if self.use_cuda:
            latent_samples = latent_samples.cuda()
        generated_data = self.G(latent_samples)
        return generated_data

    def sample(self, num_samples):
        generated_data = self.sample_generator(num_samples)
        # Remove color channel
        return generated_data.data.cpu().numpy()[:, 0, :, :]
