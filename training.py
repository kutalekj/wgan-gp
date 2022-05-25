import math

import imageio
import numpy as np
import torch
from torchvision.utils import make_grid
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
from matplotlib import pyplot as plt
from skimage.color import lab2rgb


class Trainer:
    def __init__(
            self,
            generator,
            discriminator,
            gen_optimizer,
            dis_optimizer,
            gp_weight=10,
            critic_iterations=5,
            print_every=5,
            plot_every=2,
            plot_first_n=4,
            use_cuda=False,
    ):
        self.G = generator
        self.G_opt = gen_optimizer
        self.D = discriminator
        self.D_opt = dis_optimizer
        self.losses = {"G": [], "D": [], "GP": [], "gradient_norm": []}
        self.num_steps = 0
        self.use_cuda = use_cuda
        self.gp_weight = gp_weight
        self.critic_iterations = critic_iterations
        self.print_every = print_every
        self.plot_first_n = plot_first_n
        self.plot_every = plot_every

        if self.use_cuda:
            self.G.cuda()
            self.D.cuda()

    def _critic_train_iteration(self, og_data, grayscale_data):
        # ANK: v promenne data je batch nactenych obrazku, v nasem pripade to budou barevne ground_truth obrazky protoze jsou pak pouzity v diskriminatoru

        # assert og_data.size()[0] == grayscale_data.size()[0]
        # Get generated data
        # batch_size = og_data.size()[0]

        # TODO: ANK: ultra dulezite, tady ta funkce bude potreba hodne predelat, tak aby generovala prostory ab a navic by na vstupu mela mit Lab cernobileho obrazku (s napovedou barev)
        generated_data = self.sample_generator(grayscale_data)

        # Calculate probabilities on real and generated data
        # TODO: JKU: Modify generator to concatenate input (L*a*b) with noise and output a 2-channel (*a*b) image
        og_data = Variable(og_data)
        if self.use_cuda:
            og_data = og_data.cuda()

        # ANK: hrozne dulezity krok, d_real jsou asi pravdepodobnosti ze skutecna data jsou skutecna
        # d_gen jsou pravdepodobnosti ze vygenerovana data jsou skutecna
        d_real = self.D(og_data)
        d_generated = self.D(generated_data)

        # Calculate the gradient penalty
        gradient_penalty = self._gradient_penalty(og_data, generated_data)
        self.losses["GP"].append(gradient_penalty.data)

        # Get total loss and optimize
        # https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
        self.D_opt.zero_grad()
        d_loss = d_generated.mean() - d_real.mean() + gradient_penalty
        # ANK: tomuto kroku nerozumim, vola se backward na hodnotu loss .. co se tim docili?
        d_loss.backward()

        # Perform a single optimization step
        self.D_opt.step()
        self.losses["D"].append(d_loss.data)

    def _generator_train_iteration(self, grayscale_data):
        # https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
        self.G_opt.zero_grad()

        # Get generated data
        generated_data = self.sample_generator(grayscale_data)

        # Calculate loss and optimize
        d_generated = self.D(generated_data)
        g_loss = -d_generated.mean()
        g_loss.backward()

        # Perform a single optimization step
        self.G_opt.step()
        self.losses["G"].append(g_loss.data)

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
        gradients = torch_grad(
            outputs=prob_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones(prob_interpolated.size()).cuda()
            if self.use_cuda
            else torch.ones(prob_interpolated.size()),
            create_graph=True,
            retain_graph=True,
        )[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # So, flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)
        self.losses["gradient_norm"].append(gradients.norm(2, dim=1).mean().data)

        # Derivatives of the gradient close to 0 can cause problems because of the square root
        # So, manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return self.gp_weight * ((gradients_norm - 1) ** 2).mean()

    def _train_epoch(self, og_data_loader, grayscale_data_loader):
        # A single training epoch
        # ANK: tady se nacte do promenne data vzdy cely batch dat - tj tensor o rozmerech (batch_size, channels, 256, 256)
        for i, (og_data, grayscale_data) in enumerate(zip(og_data_loader, grayscale_data_loader)):
            self.num_steps += 1
            # Discriminator training?
            self._critic_train_iteration(
                og_data, grayscale_data
            )  # TODO: JKU: A 3-channel REAL image (L*a*b) should be passed in
            # self._critic_train_iteration(data[0])

            # Debug plotting
            batch_size = og_data.size()[0]
            if i % self.plot_every == 0 and self.plot_first_n <= batch_size:
                grayscale_condition_samples = grayscale_data[0:self.plot_first_n, ...]
                sample_generated = self.sample_generator(grayscale_condition_samples)
                self.plot_first_n_times_n_batch_images(grayscale_condition_samples, color_mode='rgb', save=True,
                                                       name=f"Iter:{i} gen_before", n=int(math.sqrt(self.plot_first_n)))
                self.plot_first_n_times_n_batch_images(sample_generated, color_mode='rgb', save=True,
                                                       name=f"Iter:{i} gen_after", n=int(math.sqrt(self.plot_first_n)))

            # Only update generator every |critic_iterations| iterations (every second iteration?)
            if self.num_steps % self.critic_iterations == 0:
                # Generator training?
                self._generator_train_iteration(
                    grayscale_data
                )  # TODO: JKU: A 3-chn. USER image (L*a*b) should be passed in
                # self._generator_train_iteration(data[0])
                # TODO: JKU: Concatenate 2-chn. (*a,*b) generator output with the 1-chn. grayscale prior ('L' from REAL)

            # STDOUT print
            if i % self.print_every == 0:
                print("Iteration {}".format(i + 1))
                print("D: {}".format(self.losses["D"][-1]))
                print("GP: {}".format(self.losses["GP"][-1]))
                print("Gradient norm: {}".format(self.losses["gradient_norm"][-1]))
                if self.num_steps > self.critic_iterations:
                    print("G: {}".format(self.losses["G"][-1]))

    def train(self, og_data_loader, grayscale_data_loader, epochs):

        # Main training loop
        for epoch in range(epochs):
            print("\nEpoch {}".format(epoch + 1))
            self._train_epoch(og_data_loader, grayscale_data_loader)

    def sample_generator(self, grayscale_data):
        # "Variable" is a wrapper around a PyTorch Tensor, representing a node in a computational graph
        # latent_samples = Variable(self.G.sample_latent(num_samples))  # Get gaussian noise data
        # if self.use_cuda:
        #     latent_samples = latent_samples.cuda()
        generated_data = self.G(grayscale_data)
        return generated_data

    # def sample(self, num_samples):
    #     generated_data = self.sample_generator(num_samples)
    #     # Remove color channel
    #     return generated_data.data.cpu().numpy()[:, 0, :, :]

    @staticmethod
    def plot_first_n_times_n_batch_images(image_batch_tensor, n=2, de_norm=True, color_mode='rgb', save=False, name=''):
        fig, ax = plt.subplots(2, 2, figsize=(10, 10))
        img3 = []

        i = 0
        # (3, 256, 256, batch_size) image tensor on the input, iterate over images in batch
        for im in enumerate(image_batch_tensor):
            if i < int(n * n):
                img1 = im[1]

                # L*a*b channels de-normalization
                if de_norm:
                    img1_l = img1[[0], ...] * 50.0 + 50.0
                    img1_ab = img1[[1, 2], ...] * 110.0
                    img1 = torch.cat([img1_l, img1_ab], 0)

                # (color_channels, width, height) -> (width, height, color_channels)
                img2 = torch.permute(img1, (1, 2, 0))

                # Optional L*a*b -> RGB
                if color_mode == 'rgb':
                    img3 = lab2rgb(img2.detach().numpy())
                    ax[int(i / n), int(i % n)].imshow(img3)
                else:
                    img3 = img2
                    ax[int(i / n), int(i % n)].imshow(img3.detach().numpy())

                ax[int(i / n), int(i % n)].axis("off")
            else:
                break
            i = i + 1

        # Save and show image
        if save:
            im_name = "rgb_" + str(name) + ".png" if color_mode == 'rgb' else "lab_" + str(name) + ".png"
            fig.suptitle(im_name[:-4], fontsize=20)
            if color_mode == 'lab':
                img3 = img3.detach().numpy()
            else:  # TODO: JKU: Saving does not work for L*a*b
                fig.savefig(im_name)
        fig.show()
