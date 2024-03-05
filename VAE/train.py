# import torch
# from torch import nn
# import torch.utils.data
# from torch import nn, optim
# from torch.nn import functional as F
# import sys
# import concurrent.futures
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# import numpy as np
# import torchvision
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
# from torch.utils.data import DataLoader
# import json
# from VAE import VAE
# from VAE import Encoder
# from VAE import Decoder

# # Assuming z_dim is the dimensionality of your latent space
# input_dim = 784
# hidden_dim = 100
# latent_dims = [2, 4, 8, 16, 32, 64]
# output_dim = input_dim
# learning_rate = 1e-3
# batch_size = 256

# transform = transforms.Compose([transforms.ToTensor()])
# mnist_trainset = torchvision.datasets.MNIST(
#     root="./data", train=True, download=True, transform=transform
# )
# data_loader = DataLoader(
#     mnist_trainset, batch_size=batch_size, shuffle=True, num_workers=4
# )


# for latent_dim in tqdm(latent_dims, desc="Dim"):
#     encoder = Encoder(input_dim, hidden_dim, latent_dim)
#     decoder = Decoder(latent_dim, hidden_dim, output_dim)

#     vae_args = {
#         "encoder": encoder,
#         "decoder": decoder,
#         "optimizer": optim.Adam(
#             list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate
#         ),
#     }
#     vae = VAE(vae_args)

#     # vae.load_state_dict(torch.load(f"vae_mnist_weights_{latent_dim}.pth"))
#     losses = vae.train(data_loader, 100, batch_size, latent_dim)
#     print(losses)
#     torch.save(vae.state_dict(), f"vae_mnist_weights_{latent_dim}.pth")

import torch
from torch import nn
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import sys
import concurrent.futures
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

# Assuming z_dim is the dimensionality of your latent space
input_dim = 784
hidden_dim = 100
latent_dims = [2, 4, 8, 16, 32, 64]
output_dim = input_dim
learning_rate = 1e-3
batch_size = 256

transform = transforms.Compose([transforms.ToTensor()])
mnist_trainset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
data_loader = DataLoader(
    mnist_trainset, batch_size=batch_size, shuffle=True, num_workers=4
)


def train(latent_dim):
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

    # vae.load_state_dict(torch.load(f"vae_mnist_weights_{latent_dim}.pth"))
    losses = vae.train(data_loader, 100, batch_size, latent_dim)
    print(losses)
    torch.save(vae.state_dict(), f"vae_mnist_weights_{latent_dim}.pth")


with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(train, latent_dims))