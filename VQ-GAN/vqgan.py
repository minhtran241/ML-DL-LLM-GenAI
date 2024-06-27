import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder
from codebook import Codebook


class VQGAN(nn.Module):
    def __init__(self, args):
        super(VQGAN, self).__init__()
        self.encoder = Encoder(args).to(device=args.device)
        self.decoder = Decoder(args).to(device=args.device)
        self.codebook = Codebook(args).to(device=args.device)
        self.quant_conv = nn.Conv2d(
            args.latent_dim,
            args.latent_dim,
            kernel_size=1,
        ).to(device=args.device)
        self.post_quant_conv = nn.Conv2d(
            args.latent_dim,
            args.latent_dim,
            kernel_size=1,
        ).to(device=args.device)

    def forward(self, imgs):
        z_e = self.encoder(imgs)
        z_q = self.quant_conv(z_e)
        z_q, min_encoding_indices, loss = self.codebook(z_e)
        z_q = self.post_quant_conv(z_q)
        x_hat = self.decoder(z_q)
        return x_hat, min_encoding_indices, loss

    def encode(self, imgs):
        encoded_imgs = self.encoder(imgs)
        quant_conv_encoded_imgs = self.quant_conv(encoded_imgs)
        z_q, min_encoding_indices, loss = self.codebook(quant_conv_encoded_imgs)
        return z_q, min_encoding_indices, loss

    def decode(self, z):
        z_q = self.post_quant_conv(z)
        x_hat = self.decoder(z_q)
        return x_hat

    def calculate_lambda(self, perceptual_loss, gan_loss):
        last_layer = self.decoder.model[-1]
        last_layer_weight = last_layer.weight
        perceptual_loss_grads = torch.autograd.grad(
            perceptual_loss, last_layer_weight, retain_graph=True
        )[0]
        gan_loss_grads = torch.autograd.grad(
            gan_loss, last_layer_weight, retain_graph=True
        )[0]
        lambda_ = torch.norm(perceptual_loss_grads) / (
            torch.norm(gan_loss_grads) + 1e-4
        )
        lambda_ = torch.clamp(
            lambda_, 0, 1e4
        ).detach()  # Clipping lambda to avoid exploding gradients and making it a constant
        return lambda_ * 0.8

    @staticmethod
    def adopt_weight(disc_factor, i, threshold, value=0.0):
        """
        Adopt weight for the discriminator loss: We don't want the discriminator loss to dominate the perceptual loss so we adopt the weight of the discriminator loss to be 0.0 for the first threshold epochs. This is done to stabilize the training process.

        Args:
            disc_factor (float): The weight of the discriminator loss
            i (int): The current epoch
            threshold (int): The epoch at which the weight of the discriminator loss is adopted
            value (float, optional): The value to adopt the weight to. Defaults to 0.0.

        Returns:
            float: The adopted weight of the discriminator loss
        """
        if i < threshold:
            return value
        return disc_factor

    def load_checkpoint(self, path):
        """
        Load the model from a checkpoint file. The checkpoint file should be a .pt file.

        Args:
            path (str): The path to the checkpoint file

        Returns:
            None
        """
        self.load_state_dict(torch.load(path))
