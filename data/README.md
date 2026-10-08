Dataset
This project was trained using the CelebA (CelebFaces Attributes) face dataset.
The full dataset is not included in this repository because of its size.
Expected Structure
The training script uses PyTorch `ImageFolder`, so place the CelebA images inside one class subfolder:
```text
dataset/
└── celeba/
    ├── 000001.jpg
    ├── 000002.jpg
    ├── 000003.jpg
    └── ...
```
The class name itself is not used by the GAN; the subfolder is required only because `torchvision.datasets.ImageFolder` expects class directories.
Preprocessing
Training images are:
resized to 64 pixels;
center-cropped to `64 × 64`;
converted to tensors;
normalized to the range `[-1, 1]`.
Please obtain CelebA from its official or authorized distribution source and follow the dataset's license and usage terms
