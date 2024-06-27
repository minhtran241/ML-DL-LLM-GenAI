import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader


def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(
        torch.cat([torch.cat([i for i in images.cpu()], dim=-1)], dim=-2)
        .permute(1, 2, 0)
        .cpu()
    )
    plt.show()


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)  # make a grid of images
    ndarr = grid.permute(1, 2, 0).to("cpu").numpy()  # convert to numpy array
    im = Image.fromarray(ndarr)  # create an image from the array
    im.save(path)


def get_data(args) -> DataLoader:
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(80),
            torchvision.transforms.RandomResizedCrop(args.img_size, scale=(0.8, 1.0)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader


def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)
