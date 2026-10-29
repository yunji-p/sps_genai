import math

import matplotlib.pyplot as plt
import torch


def generate_samples(model, device, num_samples=10):
    """Sample latent points, run the generator, and plot the images on a grid."""
    generator = model.generator
    generator.to(device)
    generator.eval()

    with torch.no_grad():
        noise = torch.randn(num_samples, model.z_dim, device=device)
        images = generator(noise)

    images = (images * 0.5 + 0.5).clamp(0, 1).cpu()

    cols = math.ceil(math.sqrt(num_samples))
    rows = math.ceil(num_samples / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.3, rows * 1.3), squeeze=False)
    for i, ax in enumerate(axes.flat):
        if i < num_samples:
            ax.imshow(images[i].squeeze(0), cmap="gray")
        ax.axis("off")
    plt.tight_layout()
    plt.show()

    return images