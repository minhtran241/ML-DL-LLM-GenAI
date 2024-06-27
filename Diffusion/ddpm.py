import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch import optim
from tqdm import tqdm
import logging
from torch.utils.tensorboard import SummaryWriter
from modules import UNet, UNetConditional, EMA
from utils import plot_images, save_images, get_data, setup_logging
import numpy as np
import copy

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt="%m/%d/%Y %H:%M:%S",
)


class Diffusion:
    def __init__(
        self, noise_steps=1000, beta_min=1e-4, beta_max=0.02, img_size=64, device="cuda"
    ):
        self.noise_steps = noise_steps
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = self.alpha.cumprod(dim=0)

    def prepare_noise_schedule(self):
        betas = torch.linspace(
            self.beta_min, self.beta_max, self.noise_steps
        )  # create beta values from beta_min to beta_max
        return betas

    def noise_images(self, x, t):
        sqrt_alpha_hat = self.alpha_hat[t].sqrt()[:, None, None, None]
        sqrt_one_minus_alpha_hat = (1.0 - self.alpha_hat[t]).sqrt()[:, None, None, None]
        noise = torch.rand_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise, noise

    def sample_timesteps(self, n: int):
        """
        Sample n timesteps from the noise schedule

        Args:
            n (int): number of timesteps to sample

        Returns:
            torch.Tensor: tensor of n timesteps
        """
        return torch.randint(low=0, high=self.noise_steps, size=(n,))

    def sample(self, model: nn.Module, n: int, labels, cfg_scale=3):
        """
        Sample n images from the model

        Args:
            model (nn.Module): model to sample from
            n (int): number of images to sample

        Returns:
            torch.Tensor: tensor of n images
        """
        logging.info(f"Sampling {n} images from the model")
        model.eval()
        with torch.inference_mode():
            x = torch.randn(n, 3, self.img_size, self.img_size).to(
                self.device
            )  # random noise image
            for i in tqdm(
                reversed(range(1, self.noise_steps)), position=0
            ):  # reverse loop of time steps from 999 to 1
                t = (
                    (torch.ones(n) * i).long().to(self.device)
                )  # create tensor of time step i
                predicted_noise = model(x, t, labels)  # get predicted noise from model
                if cfg_scale > 0:
                    uncond_predicted_noise = model(
                        x, t, None
                    )  # get unconditional predicted noise
                    predicted_noise = torch.lerp(
                        uncond_predicted_noise, predicted_noise, cfg_scale
                    )  # linear interpolation between unconditional and conditional predicted noise
                alpha = self.alpha[t][
                    :, None, None, None
                ]  # get alpha value for time step i
                alpha_hat = self.alpha_hat[t][
                    :, None, None, None
                ]  # get alpha_hat value for ti`me step i
                beta = self.beta[t][
                    :, None, None, None
                ]  # get beta value for time step i
                if i > 1:
                    noise = torch.randn_like(x)  # create random noise
                else:
                    noise = torch.zeros_like(x)  # create zero noise for last time step
                x = (
                    1
                    / torch.sqrt(alpha)
                    * (
                        x
                        - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise
                    )
                    + torch.sqrt(beta) * noise
                )  # formula: x = 1/sqrt(alpha) * (x - (1-alpha)/sqrt(1-alpha_hat) * predicted_noise) + sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2  # -> (0, 2)  # -> (0, 1)
        x = (x * 255).type(torch.uint8)  # scale to 0 and 255 (valid pixel values)
        return x


def train(args):
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    model = UNetConditional(num_classes=args.num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    difussion = Diffusion(
        img_size=args.image_size,
        device=device,
    )
    logger = SummaryWriter(log_dir=os.path.join("runs", args.run_name))
    l = len(dataloader)
    ema = EMA(beta=0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    for epoch in range(args.epochs):
        logging.info(f"Epoch {epoch + 1}/{args.epochs}")
        pbar = tqdm(dataloader)
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            t = difussion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = difussion.noise_images(images, t)
            if np.random.rand() < 0.1:
                labels = None
            predicted_noise = model(x_t, t, labels)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        if epoch % 10 == 0:
            labels = torch.arange(10).long().to(device)
            sampled_images = difussion.sample(model, n=images.shape[0], labels=labels)
            ema_sampled_images = difussion.sample(
                ema_model, n=images.shape[0], labels=labels
            )
            plot_images(sampled_images)
            save_images(sampled_images, f"results/{args.run_name}/sampled_{epoch}.png")
            save_images(
                ema_sampled_images, f"results/{args.run_name}/ema_sampled_{epoch}.png"
            )
            torch.save(model.state_dict(), f"models/{args.run_name}/model_{epoch}.pt")
            torch.save(
                ema_model.state_dict(), f"models/{args.run_name}/ema_model_{epoch}.pt"
            )
            torch.save(
                optimizer.state_dict(), f"models/{args.run_name}/optimizer_{epoch}.pt"
            )
