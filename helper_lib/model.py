import torch.nn as nn
from torchvision.models import resnet18

class Generator(nn.Module):
    """Noise (B, 100) to an MNIST-shaped image (B, 1, 28, 28) in [-1, 1]."""

    def __init__(self, z_dim=100):
        super().__init__()
        self.z_dim = z_dim
        self.fc = nn.Linear(z_dim, 7 * 7 * 128)
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU(inplace=True)
        self.deconv2 = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, z):
        x = self.fc(z)                              # (B, 6272)
        x = x.view(x.size(0), 128, 7, 7)            # (B, 128, 7, 7)
        x = self.act1(self.bn1(self.deconv1(x)))    # (B, 64, 14, 14)
        return self.tanh(self.deconv2(x))           # (B, 1, 28, 28)

class Discriminator(nn.Module):
    """An image (B, 1, 28, 28) to a single real/fake score (B, 1)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * 7 * 7, 1)

    def forward(self, x):
        x = self.act1(self.conv1(x))                # (B, 64, 14, 14)
        x = self.act2(self.bn2(self.conv2(x)))      # (B, 128, 7, 7)
        return self.fc(self.flatten(x))             # (B, 1)

class GAN(nn.Module):
    """Holds both networks so the trainer can take a single model object."""

    def __init__(self, z_dim=100):
        super().__init__()
        self.z_dim = z_dim
        self.generator = Generator(z_dim)
        self.discriminator = Discriminator()

    def forward(self, z):
        return self.generator(z)
    
def get_model(model_name):
    """Return a model instance by name."""
    models = {
        "GAN": GAN,
    }
    if model_name not in models:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {sorted(models)}"
        )
    return models[model_name]()
    