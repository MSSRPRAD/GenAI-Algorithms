import torch
from torch import nn
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import json
from VAE import VAE
from VAE import Encoder
from VAE import Decoder


def plotter(images, name):

    # Create a subplot for visualizing all generated images
    fig, axs = plt.subplots(2, 3, figsize=(10, 8))
    fig.tight_layout(pad=0.1)

    i = 0

    for image in images:
        # Decode each random_z to get the generated image
        generated_image = image

        # print(generated_image.shape)
        # Plot the generated image in the subplot
        row, col = divmod(i, 3)
        axs[row, col].imshow(generated_image.reshape(28, 28), cmap="gray")
        axs[row, col].axis("off")
        i += 1

    # Save the figure with all generated images
    plt.savefig(name + ".png")


transform = transforms.Compose([transforms.ToTensor()])
mnist_trainset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)

X = mnist_trainset.data.reshape(60000, 784)
# indexes = torch.randperm(X.shape[0])
# original = X[indexes][:6]
original = X[:6]
original = torch.tensor(original, dtype=torch.float)

plotter(original.detach().numpy() / 255, "original")

# Set device to CPU
device = torch.device("cpu")

# Assuming z_dim is the dimensionality of your latent space
input_dim = 784
hidden_dim = 100
latent_dims = [2, 4, 8, 16, 32, 64]
output_dim = input_dim
learning_rate = 1e-3
batch_size = 256

logs = []
for latent_dim in tqdm(latent_dims, desc="Dim"):

    encoder = Encoder(input_dim, hidden_dim, latent_dim)
    decoder = Decoder(latent_dim, hidden_dim, output_dim)

    vae_args = {
        "encoder": encoder,
        "decoder": decoder,
        "optimizer": optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate
        ),
    }
    vae = VAE(vae_args)
    vae.load_state_dict(
        torch.load(
            f"vae_mnist_weights_{latent_dim}.pth", map_location=torch.device("cpu")
        )
    )

    reconstructed, mean, var = vae(original / 255)
    # reconstructed *= 255
    MSE = (reconstructed.detach().numpy() * 255 - original.detach().numpy()) ** 2
    MSE = np.mean(MSE)

    logs.append((latent_dim, MSE.item()))

    # Plot Reconstructed
    plotter(reconstructed.detach().numpy(), f"reconstructed_{latent_dim}")

json.dump(logs, open("logs.json", "w"))
