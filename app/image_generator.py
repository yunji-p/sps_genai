import io
from pathlib import Path

import torch
from PIL import Image
from torchvision.utils import make_grid

from helper_lib.model import Generator

DEFAULT_WEIGHTS = Path(__file__).parent / "weights" / "gan_mnist_generator.pth"


class DigitGenerator:
    """Loads the trained generator once and turns noise into PNG bytes."""

    def __init__(self, weights_path: Path = DEFAULT_WEIGHTS, device: str = "cpu"):
        if not Path(weights_path).exists():
            raise FileNotFoundError(f"No generator weights at {weights_path}")
        self.device = torch.device(device)
        self.model = Generator().to(self.device)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()

    @torch.no_grad()
    def generate(self, num_images: int = 16, seed: int | None = None) -> torch.Tensor:
        if seed is not None:
            torch.manual_seed(seed)
        noise = torch.randn(num_images, self.model.z_dim, device=self.device)
        images = self.model(noise)
        return (images * 0.5 + 0.5).clamp(0, 1).cpu()

    def generate_png(self, num_images: int = 16, seed: int | None = None,
                     scale: int = 4) -> io.BytesIO:
        images = self.generate(num_images, seed)
        nrow = max(1, int(num_images ** 0.5))
        grid = make_grid(images, nrow=nrow, padding=2, pad_value=1.0)
        array = (grid.permute(1, 2, 0).numpy() * 255).round().astype("uint8")
        pil = Image.fromarray(array)
        if scale > 1:                        # 28px tiles are tiny in a browser
            pil = pil.resize((pil.width * scale, pil.height * scale), Image.NEAREST)
        buffer = io.BytesIO()
        pil.save(buffer, format="PNG")
        buffer.seek(0)
        return buffer