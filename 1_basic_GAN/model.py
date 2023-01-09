"""
Basic Fully connected GAN network builder
"""
import warnings
warnings.filterwarnings("ignore")
import pytest
import pdb
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter 


### Discriminator Model: D in the paper
class Discriminator(nn.Module):
    def __init__(self, in_features, hidden_size=128):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return self.disc(x)

### Generator Model: G in the paper
class Generator(nn.Module):
    def __init__(self, z_dim, img_dim,hidden_size=256):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, hidden_size),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_size, img_dim),
            nn.Tanh(),  
        )
    def forward(self, x):
        return self.gen(x)

def model_trainer(data_loader,learning_rate, im_dim,z_dim,num_epochs,batch_size,):
    image_dim = im_dim[0]*im_dim[1]*im_dim[2]
    n_channels = im_dim[2]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    disc = Discriminator(image_dim).to(device)
    gen = Generator(z_dim, image_dim).to(device)
    fixed_noise = torch.randn((batch_size, z_dim)).to(device)
    opt_disc = optim.Adam(disc.parameters(), lr=learning_rate)
    opt_gen = optim.Adam(gen.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()
    writer_fake = SummaryWriter(f"logs/fake")
    writer_real = SummaryWriter(f"logs/real")
    step = 0
    for epoch in range(num_epochs):
        for batch_idx, (real, _) in enumerate(data_loader):
            real = real.view(-1, image_dim).to(device)
            ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
            noise = torch.randn(batch_size, z_dim).to(device)
            fake = gen(noise) # generated *fake* image
            disc_real = disc(real).view(-1) # discriminator output for real images
            lossD_real = criterion(disc_real, torch.ones_like(disc_real)) # discriminator loss for real images
            disc_fake = disc(fake).view(-1) # discriminator output for fake images
            lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake)) # discriminator loss for fake images
            lossD = (lossD_real + lossD_fake) / 2 # loss== average of the two losses for real and fake images
            disc.zero_grad()
            lossD.backward(retain_graph=True) # (retain_graph=True) ==> used in order to reuse fake tensor
            opt_disc.step()
            ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
            # where the second option of maximizing doesn't suffer from
            # saturating gradients
            output = disc(fake).view(-1)
            lossG = criterion(output, torch.ones_like(output))
            gen.zero_grad()
            lossG.backward()
            opt_gen.step()
            # now outputing some images and metrics
            if batch_idx == 0:
                print(
                    f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(data_loader)} \
                        Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
                )

            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, n_channels, im_dim[0], im_dim[1])
                data = real.reshape(-1, n_channels, im_dim[0], im_dim[1])
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                writer_fake.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=step
                )
                writer_real.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )
                step += 1
 
def test_model_builders():
    """
    Test Module for diffrent models
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ### Discriminator Test: 
    image_dim = 28 * 28 * 1  
    hidden_size=128
    disc_model = Discriminator(image_dim,hidden_size=hidden_size).to(device)
    print('Discriminator Architecture:')
    print(disc_model)
    x =  torch.rand(5,image_dim)
    y = disc_model(x)
    print(y)
    print(y.shape)
    ### Generator Test:
    z_dim = 64
    hidden_size=256
    gen_model = Generator(z_dim, image_dim,hidden_size=hidden_size).to(device)
    print('Generator Architecture:')
    print(gen_model)
    x = torch.rand(5, z_dim)
    y=gen_model(x)
    print(y)
    print(y.shape)


