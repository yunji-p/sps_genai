"""Part 2: train AssignmentCNN on CIFAR-10 and save the weights the API loads.

Run from the repo root:  python -m training.train_cnn --epochs 10
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader

from app.cnn_model import AssignmentCNN, PREPROCESS

WEIGHTS_PATH = Path("app/weights/cnn_cifar10.pth")


def get_device() -> torch.device:
    if torch.cuda.is_available():          # Colab GPU
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # Apple Silicon Mac
        return torch.device("mps")
    return torch.device("cpu")


def main(epochs: int, batch_size: int, lr: float) -> None:
    torch.manual_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    train_data = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=PREPROCESS)
    test_data = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=PREPROCESS)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)

    model = AssignmentCNN().to(device)
    criterion = nn.CrossEntropyLoss()  # takes raw logits, applies softmax inside
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
        print(f"Epoch {epoch + 1}/{epochs}  loss={running_loss / len(train_loader):.4f}  "
              f"train_acc={correct / total:.3f}")

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            correct += (model(images).argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    print(f"Test accuracy: {100 * correct / total:.2f}%")

    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    cpu_weights = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save(cpu_weights, WEIGHTS_PATH)
    print(f"Saved weights to {WEIGHTS_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()
    main(args.epochs, args.batch_size, args.lr)
