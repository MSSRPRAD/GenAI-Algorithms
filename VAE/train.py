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

from VAE import VAE
from VAE import Encoder
from VAE import Decoder

input_dim = 784
hidden_dim = 512
latent_dim = 10
output_dim = input_dim
learning_rate = 1e-3
batch_size = 256

transform = transforms.Compose([transforms.ToTensor()])
mnist_trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
data_loader = DataLoader(mnist_trainset, batch_size=batch_size, shuffle=True, num_workers=4)

encoder = Encoder(input_dim, hidden_dim, latent_dim)
decoder = Decoder(latent_dim, hidden_dim, output_dim)

vae_args = {'encoder': encoder, 'decoder': decoder,
            'optimizer': optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)}
vae = VAE(vae_args)

# vae.load_state_dict(torch.load('best_vae_mnist_weights.pth'))
losses = vae.train(data_loader, 1000, batch_size)
torch.save(vae.state_dict(), 'vae_mnist_weights.pth')
plt.plot(range(len(losses)), losses, label='Training Loss')
plt.xlabel('Batch or Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Batches or Epochs')
plt.legend()
plt.savefig("losses.jpg")