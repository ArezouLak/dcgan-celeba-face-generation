Results
This folder contains face samples generated during DCGAN training.
The same fixed set of latent vectors is passed through the generator after each epoch. This makes it possible to visually track how the quality and structure of generated faces change as training progresses.
Generated samples are stored in:
```text
results/generated_samples/
```
The files are named by epoch, for example:
```text
epoch_0001.png
epoch_0010.png
epoch_0020.png
```
