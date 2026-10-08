from pathlib import Path
import argparse

import torch
from torchvision.utils import save_image

from network import Generator


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = Generator(latent_dim=args.latent_dim).to(device)
    generator.load_state_dict(torch.load(args.weights, map_location=device))
    generator.eval()

    noise = torch.randn(args.num_images, args.latent_dim, 1, 1, device=device)

    with torch.no_grad():
        generated = generator(noise).cpu()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    save_image(
        generated,
        output,
        normalize=True,
        value_range=(-1, 1),
        nrow=args.nrow,
    )

    print(f"Saved generated faces to: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="models/generator.pth")
    parser.add_argument("--output", default="results/generated_faces.png")
    parser.add_argument("--num-images", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--nrow", type=int, default=8)
    main(parser.parse_args())
