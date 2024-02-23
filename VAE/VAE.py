import torch
from torch import nn
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import sys
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from tqdm import tqdm

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

class VAE(nn.Module):
    def __init__(self, args):
        super(VAE, self).__init__()
        self.encoder = args['encoder']
        self.decoder = args['decoder']
        self.optimizer = args['optimizer']
        self.losses = []

    def calculate_loss(self, actual_x, reconstructed_x, mean, var):
        reconstruction_loss = F.binary_cross_entropy(reconstructed_x.T, actual_x, reduction='sum')
        
        kl_divergence = - 0.5 * torch.mean(1 + var - mean.pow(2) - var.exp())
        
        return reconstruction_loss + kl_divergence

    def train(self, data_loader, epochs, batch_size):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        best_loss = float('inf')
        for epoch in tqdm(range(epochs), desc="Training", unit="epoch"):
            total_loss = 0
            idx = 0
            for batch_idx, (x, _) in tqdm(enumerate(data_loader), desc="Epoch {}".format(epoch), unit="batch", leave=False):
                
                actual_x = x.view(x.size(0), -1) / 255
                actual_x = actual_x.to(torch.float32).to(device)

                self.optimizer.zero_grad()
                mean, var = self.encoder.forward(actual_x)

                var = torch.exp(0.5 * var)

                sampled_z = torch.randn_like(mean) * var + mean

                reconstructed_x = self.decoder.forward(sampled_z)

                loss = self.calculate_loss(actual_x.T, reconstructed_x, mean, var)
                loss.backward()

                total_loss += loss.item()
            
                if idx % 100 == 0:
                    print("\n-----------------")
                    print(loss.item())
                    print("-----------------")

                # print("\n=====================\n")
                # print("Batch : ")
                # print(idx)
                # print(x.shape)
                # print(actual_x.shape)
                # print(sampled_z.shape)
                # print(reconstructed_x.shape)
                # print("\n=====================\n")


                idx += 1
                self.optimizer.step()
                self.losses.append(total_loss/batch_size)

            print("====> Epoch: {} Loss: {:.4f}".format(epoch, total_loss))
            if total_loss < best_loss:
                total_loss = best_loss
                torch.save(self.state_dict(), 'best_vae_mnist_weights.pth')
        return self.losses


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim=50):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        self.latent_dim = latent_dim
        self.LeakyReLU = nn.ReLU()
        # self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.LeakyReLU(self.fc1(x))
        x = self.LeakyReLU(self.fc2(x))
        mean = self.fc_mean(x)
        var = self.fc_var(x)
        return mean, var


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.LeakyReLU = nn.ReLU()
        # self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, z):
        z = self.LeakyReLU(self.fc1(z))
        z = self.LeakyReLU(self.fc2(z))
        reconstructed_x = torch.sigmoid(self.fc3(z))
        return reconstructed_x