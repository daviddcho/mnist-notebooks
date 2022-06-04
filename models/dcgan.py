#!/usr/bin/env python3
import torch 
import torch.nn as nn

class Generator(nn.Module):
  def __init__(self, z_dim=100):
    super(Generator, self).__init__()
    C = 28
    self.layers = nn.Sequential(
      nn.ConvTranspose2d(z_dim, C*4, 4, 1, 0, bias=False),
      nn.BatchNorm2d(C*4), 
      nn.ReLU(True),
      nn.ConvTranspose2d(C*4, C*2, 3, 2, 1, bias=False),
      nn.BatchNorm2d(C*2), 
      nn.ReLU(True),
      nn.ConvTranspose2d(C*2, C, 4, 2, 1, bias=False),
      nn.BatchNorm2d(C), 
      nn.ReLU(True),
      nn.ConvTranspose2d(C, 1, 4, 2, 1, bias=False),
      nn.Tanh(),
    )

  def forward(self, x):
    return self.layers(x)

class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()
    C = 28
    self.layers = nn.Sequential(
      nn.Conv2d(1, C, 4, 2, 1, bias=False),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Conv2d(C, C*2, 4, 2, 1, bias=False),
      nn.BatchNorm2d(C*2),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Conv2d(C*2, C*4, 3, 2, 1, bias=False),
      nn.BatchNorm2d(C*4),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Conv2d(C*4, 1, 4, 1, 0, bias=False),
      nn.Sigmoid(),
    )

  def forward(self, x):
    return self.layers(x).view(-1)

class PrintLayer(nn.Module):
  def __init__(self):
    super(PrintLayer, self).__init__()

  def forward(self, x):
    print(x.shape)
    return x

if __name__ == "__main__":
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  gen = Generator().to(device)
  dis = Discriminator().to(device)

  x = torch.randn(8, 100, 1, 1).to(device)
  out = gen(x)
  out = dis(out)
