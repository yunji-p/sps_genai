"""Part 1: the CNN from the Assignment 2 spec, plus the preprocessing it expects."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

IMAGE_SIZE = 64  # the spec says the input is a 64x64 RGB image

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# CIFAR-10 images are 32x32, so we resize them up to 64x64 to match the spec.
# Training and the API both use this exact pipeline, so the model always sees
# images prepared the same way.
PREPROCESS = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),  # HxWxC 0-255  ->  CxHxW 0.0-1.0
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                         std=(0.2470, 0.2435, 0.2616)),
])


class AssignmentCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # Conv2D, 16 filters, 3x3, stride 1, padding 1:  3x64x64 -> 16x64x64
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        # MaxPool 2x2, stride 2 halves height and width (reused after each conv)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Conv2D, 32 filters, 3x3, stride 1, padding 1: 16x32x32 -> 32x32x32
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        # After the second pool: 32 channels x 16 x 16 = 8192 numbers per image
        self.fc1 = nn.Linear(32 * 16 * 16, 100)
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))  # B x 16 x 32 x 32
        x = self.pool(F.relu(self.conv2(x)))  # B x 32 x 16 x 16
        x = torch.flatten(x, start_dim=1)     # B x 8192
        x = F.relu(self.fc1(x))               # B x 100
        return self.fc2(x)                    # B x 10 raw scores (logits)
