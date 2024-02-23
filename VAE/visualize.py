import torch
import numpy as np
from torch import nn, optim
from VAE import VAE, Encoder, Decoder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Set device to CPU
device = torch.device("cpu")

# Assuming z_dim is the dimensionality of your latent space
z_dim = 10
input_dim = 784
hidden_dim = 512
latent_dim = z_dim
output_dim = input_dim
learning_rate = 1e-2

# Generate 20 random z vectors
num_samples = 15

# Generate random z vectors with increased variance
random_z = []
mean = torch.zeros(z_dim)
var = torch.ones(z_dim)
for i in range(num_samples):
    random_z.append(torch.randn_like(mean) * var * 0.2 + mean)

random_z = torch.as_tensor(np.array(random_z))

# Send the random_z to the same device as your model
random_z = random_z.to(device)


encoder = Encoder(input_dim, hidden_dim, latent_dim)
decoder = Decoder(latent_dim, hidden_dim, output_dim)

vae_args = {'encoder': encoder, 'decoder': decoder,
            'optimizer': optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)}
vae = VAE(vae_args)
vae.load_state_dict(torch.load('best_vae_mnist_weights.pth'))

# Create a subplot for visualizing all generated images
fig, axs = plt.subplots(5, 3, figsize=(10, 8))
fig.tight_layout(pad=0.1)

for i in range(num_samples):
    # Decode each random_z to get the generated image
    generated_image = 255 * vae.decoder.forward(random_z[i].unsqueeze(0))  # Unsqueeze to add batch dimension

    # Plot the generated image in the subplot
    row, col = divmod(i, 3)
    axs[row, col].imshow(generated_image.view(28, 28).detach().numpy(), cmap='gray')
    axs[row, col].axis('off')

# Save the figure with all generated images
plt.savefig('generated_images_grid.png')
