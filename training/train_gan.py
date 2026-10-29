import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from helper_lib.model import get_model
from helper_lib.trainer import train_gan

WEIGHTS_PATH = "app/weights/gan_mnist_generator.pth"


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main(epochs=20, batch_size=128, lr=2e-4):
    device = pick_device()
    print("training on:", device)

    transform = transforms.Compose([
        transforms.ToTensor(),                   # [0, 1]
        transforms.Normalize([0.5], [0.5]),      # [-1, 1], matches Tanh
    ])
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=2, drop_last=True)

    gan = get_model("GAN")
    criterion = nn.BCEWithLogitsLoss()
    opt_gen = torch.optim.Adam(gan.generator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_disc = torch.optim.Adam(gan.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    train_gan(gan, loader, criterion, (opt_gen, opt_disc), device=device, epochs=epochs)

    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
    torch.save(gan.generator.state_dict(), WEIGHTS_PATH)
    print("saved", WEIGHTS_PATH)


if __name__ == "__main__":
    main()