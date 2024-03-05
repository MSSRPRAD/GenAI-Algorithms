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
import torch
import json

matplotlib.use("Agg")


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

# Assuming z_dim is the dimensionality of your latent space
z_dims = [2, 4, 8, 16, 32, 64]
input_dim = 784

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

# original = X[:6]

indexes = torch.randperm(X.shape[0])

original = X[indexes][:6]

plotter(original, f"original")

logs = []

for z_dim in tqdm(z_dims, desc="Dim"):
    V = eigenvectors[:, :z_dim]

    reconstructed = ((original - X_mean) @ V) @ V.T + X_mean

    MSE = (reconstructed.reshape(6, 784) - original) ** 2
    MSE = np.mean(MSE)

    logs.append((z_dim, MSE))

    plotter(reconstructed, f"reconstructed_{z_dim}")

json.dump(logs, open("logs.json", "w"))
