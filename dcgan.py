#!/usr/bin/env python3
import torch 
from torch import optim, nn
import torch.nn.functional as F

class Generator(nn.Module):
  def __init__(self, z_dim=100):
    super(Generator, self).__init__()
    self.layers = nn.Sequential(
      nn.ConvTranspose2d(z_dim, 28*4, 4, 1, 0, bias=False),
      nn.BatchNorm2d(28*4), 
      nn.ReLU(True),
      nn.ConvTranspose2d(28*4, 28*2, 3, 2, 1, bias=False),
      nn.BatchNorm2d(28*2), 
      nn.ReLU(True),
      nn.ConvTranspose2d(28*2, 28, 4, 2, 1, bias=False),
      nn.BatchNorm2d(28), 
      nn.ReLU(True),
      nn.ConvTranspose2d(28, 1, 4, 2, 1, bias=False),
      nn.Tanh(),
      # n_ch x 28 x 28
    )

  def forward(self, x):
    return self.layers(x)

class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()
    self.layers = nn.Sequential(
      # n_channels x 28 x 28
      nn.Conv2d(1, 28, 4, 2, 1, bias=False),
      nn.LeakyReLU(0.2, inplace=True),

      nn.Conv2d(28, 28*2, 4, 2, 1, bias=False),
      nn.BatchNorm2d(28*2),
      nn.LeakyReLU(0.2, inplace=True),

      nn.Conv2d(28*2, 28*4, 3, 2, 1, bias=False),
      nn.BatchNorm2d(28*4),
      nn.LeakyReLU(0.2, inplace=True),

      nn.Conv2d(28*4, 1, 4, 1, 0, bias=False),
      nn.Sigmoid(),
    )

  def forward(self, x):
    return self.layers(x).view(-1)

class PrintLayer(nn.Module):
  def __init__(self):
    super(PrintLayer, self).__init__()
  def forward(self, x):
    print(f"within layers, {x.shape}")
    return x

if __name__ == "__main__":
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  gen = Generator().to(device)
  dis = Discriminator().to(device)