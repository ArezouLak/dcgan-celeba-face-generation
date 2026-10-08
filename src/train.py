from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image

from network import Generator, Discriminator


def weights_init(module):
    """DCGAN initialization: N(0, 0.02) for conv, N(1, 0.02) for batch norm."""
    name = module.__class__.__name__

    if "Conv" in name:
        nn.init.normal_(module.weight.data, 0.0, 0.02)

    elif "BatchNorm" in name:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0)


def main(args):
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    model_dir = Path(args.model_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose(
        [
            transforms.Resize(args.image_size),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    dataset = datasets.ImageFolder(root=str(data_dir), transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Training images: {len(dataset)}")

    # Save a sample of real training images.
    real_batch = next(iter(loader))[0][:64]
    save_image(
        real_batch,
        output_dir / "training_images.png",
        normalize=True,
        value_range=(-1, 1),
        nrow=8,
    )

    generator = Generator(latent_dim=args.latent_dim).to(device)
    discriminator = Discriminator(image_channels=3).to(device)

    generator.apply(weights_init)
    discriminator.apply(weights_init)

    optimizer_g = Adam(
        generator.parameters(), lr=args.lr, betas=(args.beta1, 0.999)
    )
    optimizer_d = Adam(
        discriminator.parameters(), lr=args.lr, betas=(args.beta1, 0.999)
    )

    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, args.latent_dim, 1, 1, device=device)

    g_losses = []
    d_losses = []

    for epoch in range(args.epochs):
        for batch_index, (real_images, _) in enumerate(loader):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            # ---- Train discriminator ----
            discriminator.zero_grad()

            real_labels = torch.ones(batch_size, device=device)
            fake_labels = torch.zeros(batch_size, device=device)

            real_output = discriminator(real_images).view(-1)
            d_loss_real = criterion(real_output, real_labels)
            d_loss_real.backward()

            noise = torch.randn(
                batch_size, args.latent_dim, 1, 1, device=device
            )
            fake_images = generator(noise)

            fake_output = discriminator(fake_images.detach()).view(-1)
            d_loss_fake = criterion(fake_output, fake_labels)
            d_loss_fake.backward()

            d_loss = d_loss_real + d_loss_fake
            optimizer_d.step()

            # ---- Train generator ----
            generator.zero_grad()

            output = discriminator(fake_images).view(-1)
            g_loss = criterion(output, real_labels)
            g_loss.backward()
            optimizer_g.step()

            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())

            if batch_index % 50 == 0:
                print(
                    f"[{epoch + 1}/{args.epochs}] "
                    f"[{batch_index}/{len(loader)}] "
                    f"Loss_D={d_loss.item():.4f} "
                    f"Loss_G={g_loss.item():.4f} "
                    f"D(x)={real_output.mean().item():.4f} "
                    f"D(G(z))={fake_output.mean().item():.4f}"
                )

        # Generate the same fixed latent vectors after every epoch.
        generator.eval()
        with torch.no_grad():
            fake = generator(fixed_noise).cpu()
        generator.train()

        save_image(
            fake,
            output_dir / f"epoch_{epoch + 1:04d}.png",
            normalize=True,
            value_range=(-1, 1),
            nrow=8,
        )

    # Save model state dictionaries.
    torch.save(generator.state_dict(), model_dir / "generator.pth")
    torch.save(discriminator.state_dict(), model_dir / "discriminator.pth")

    # Save GAN loss curve.
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(g_losses, label="Generator")
    plt.plot(d_losses, label="Discriminator")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "gan_loss.png")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="dataset")
    parser.add_argument("--output-dir", default="results/generated_samples")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=2)
    main(parser.parse_args())
