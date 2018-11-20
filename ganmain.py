# import torch
# from torch.utils.data import DataLoader
# from torchvision import transforms
# from torchvision.datasets import ImageFolder
# import numpy as np
# import matplotlib.pyplot as plt
# plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
# plt.rcParams['image.interpolation'] = 'nearest'
# plt.rcParams['image.cmap'] = 'gray'
# from gan.train import train
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# from gan.losses import discriminator_loss, generator_loss
# from gan.losses import ls_discriminator_loss, ls_generator_loss
# from gan.models import Discriminator, Generator

# batch_size = 128
# scale_size = 64  # We resize the images to 64x64 for training

# celeba_root = 'celeba_data'

# celeba_train = ImageFolder(root=celeba_root, transform=transforms.Compose([
#   transforms.Resize(scale_size),
#   transforms.ToTensor(),
# ]))

# celeba_loader_train = DataLoader(celeba_train, batch_size=batch_size, drop_last=True)
# imgs = celeba_loader_train.__iter__().next()[0].numpy().squeeze()
# show_images(imgs, color=True)
# NOISE_DIM = 100
# NUM_EPOCHS = 20
# learning_rate = 0.0002


import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
import sys
sys.path.append('/home/jkane021/gan/gan')
from gan.train import train
from gan.utils import sample_noise, show_images, deprocess_img, preprocess_img
from gan.losses import discriminator_loss, generator_loss, ls_discriminator_loss, ls_generator_loss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NOISE_DIM = 100
batch_size = 128

mnist = datasets.MNIST('./MNIST_data', train=True, download=True,
                           transform=transforms.ToTensor())
loader_train = DataLoader(mnist, batch_size=batch_size, drop_last=True)


imgs = loader_train.__iter__().next()[0].view(batch_size, 784).numpy().squeeze()
show_images(imgs)

class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size() # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

def discriminator():
    """
    Initialize and return a simple discriminator model.
    """
    model = torch.nn.Sequential( Flatten(),
                                torch.nn.Linear(784, 256), 
                                torch.nn.LeakyReLU(),
                                torch.nn.Linear(256, 256), 
                                torch.nn.LeakyReLU(),
                                torch.nn.Linear(256, 1)
    )
    return model

def generator(noise_dim=NOISE_DIM):
    """
    Initialize and return a simple generator model.
    """
    
    model = nn.Sequential(
        torch.nn.Linear(noise_dim, 1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, 1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, 784),
        torch.nn.Tanh()
    )

    return model

def main():
	D = discriminator().to(device)
	G = generator().to(device)

	D_optimizer = torch.optim.Adam(D.parameters(), lr=1e-3, betas = (0.5, 0.999))
	G_optimizer = torch.optim.Adam(G.parameters(), lr=1e-3, betas = (0.5, 0.999))

	train(D, G, D_optimizer, G_optimizer, discriminator_loss, generator_loss, train_loader=loader_train, num_epochs=10, device=device)


if __name__ == "__main__":
	main()