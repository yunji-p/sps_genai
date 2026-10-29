import torch

def train_gan(model, data_loader, criterion, optimizer, device="cpu", epochs=10):
    """Train a GAN. `optimizer` is a (generator_optimizer, discriminator_optimizer) pair."""
    opt_gen, opt_disc = optimizer
    model.to(device)
    model.train()
    generator, discriminator = model.generator, model.discriminator

    for epoch in range(1, epochs + 1):
        d_running = g_running = 0.0

        for real, _ in data_loader:
            real = real.to(device)
            batch = real.size(0)
            ones = torch.ones(batch, 1, device=device)
            zeros = torch.zeros(batch, 1, device=device)

            # 1. discriminator: score real high, score fake low
            noise = torch.randn(batch, model.z_dim, device=device)
            fake = generator(noise)
            loss_d = criterion(discriminator(real), ones) \
                   + criterion(discriminator(fake.detach()), zeros)
            opt_disc.zero_grad()
            loss_d.backward()
            opt_disc.step()

            # 2. generator: label its own fakes as real
            loss_g = criterion(discriminator(fake), ones)
            opt_gen.zero_grad()
            loss_g.backward()
            opt_gen.step()

            d_running += loss_d.item()
            g_running += loss_g.item()

        batches = len(data_loader)
        print(f"epoch {epoch:2d}/{epochs}  "
              f"D {d_running / batches:.4f}  G {g_running / batches:.4f}", flush=True)

    return model