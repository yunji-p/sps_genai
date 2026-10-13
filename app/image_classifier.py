import io
from pathlib import Path

import torch
from PIL import Image

from app.cnn_model import AssignmentCNN, CIFAR10_CLASSES, PREPROCESS

WEIGHTS_PATH = Path(__file__).parent / "weights" / "cnn_cifar10.pth"


class ImageClassifier:
    def __init__(self, weights_path: Path = WEIGHTS_PATH):
        self.model = AssignmentCNN(num_classes=len(CIFAR10_CLASSES))
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict(self, image_bytes):
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        x = PREPROCESS(image).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0]
        top = int(torch.argmax(probs))
        return {
            "predicted_class": CIFAR10_CLASSES[top],
            "confidence": round(float(probs[top]), 4),
            "probabilities": {
                name: round(float(p), 4) for name, p in zip(CIFAR10_CLASSES, probs)
            },
        }