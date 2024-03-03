import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

matplotlib.use("Agg")


def plotter(images, name):

    # Create a subplot for visualizing all generated images
    fig, axs = plt.subplots(5, 3, figsize=(10, 8))
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

# Assuming z_dim is the dimensionality of your latent space
z_dim = 2
input_dim = 784
sigma = 10

X = mnist_trainset.data

X = X.numpy().reshape(X.shape[0], -1)

X_mean = np.mean(X, axis=0).reshape(784, 1).T

X_std = X - X_mean

# print(X_std.shape)

# Calculate the covariance matrix
cov_matrix = np.cov(X_std, rowvar=False)

# Calculate the eigenvectors and eigenvalues
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# Sort eigenvalues and corresponding eigenvectors in descending order
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

R = np.identity(z_dim)
Um = eigenvectors[:, :z_dim]
Lm = np.diag(eigenvalues[:z_dim])
Wml = Um @ np.sqrt(Lm - (sigma**2) * np.identity(z_dim)) @ R
M = Wml.T @ Wml + (sigma**2) * np.identity(z_dim)

print(M)

original = X[:15]


z = np.random.normal(0, 1, (15, z_dim))
print(z.shape)
print("================")

print(z.shape, Wml.shape, X_mean.shape)

original = z @ Wml.T + X_mean

plotter(original, "original")

# original = X[:15]

# plotter(original, "original")

# reconstructed = ((original - X_mean) @ V) @ V.T + X_mean

# plotter(reconstructed, "reconstructed")
