#!/usr/bin/env python3
import torch
import torch.nn as nn
#from dcgan import PrintLayer

class VAE(nn.Module):
  def __init__(self, zdim=20):
    super(VAE, self).__init__()
    self.conv = nn.Sequential(
      #PrintLayer(),
      nn.Conv2d(1, 64, 3, 2, 1),
      nn.BatchNorm2d(64), 
      nn.ReLU(),
      #PrintLayer(),
      nn.Conv2d(64, 128, 3, 2, 1),
      nn.BatchNorm2d(128),
      nn.ReLU(),
      #PrintLayer(),
    )
    self.fc1 = nn.Linear(6272, zdim)

    self.fc2 = nn.Sequential(
      nn.Linear(zdim, 64*7*7*2),
      nn.BatchNorm1d(64*7*7*2),
      nn.ReLU(),
    )
    self.conv_transpose = nn.Sequential(
      nn.ConvTranspose2d(2*7*7*64, 128, 5, 2), 
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.ConvTranspose2d(128, 64, 5, 2), 
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.ConvTranspose2d(64, 1, 4, 2),
      nn.Sigmoid(),
    )

  def encoder(self, x):
    x = self.conv(x)
    x = x.reshape(x.shape[0], -1)
    x = self.fc1(x)
    return x
 
  def decoder(self, x):
    x = self.fc2(x)
    x = x.reshape(x.shape[0], -1, 1, 1)
    x = self.conv_transpose(x)
    return x

  def forward(self, x):
    z = self.encoder(x)
    return self.decoder(z)

if __name__ == "__main__":
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  vae = VAE().to(device)
  x = torch.rand(4, 1, 28, 28)
  out = vae(x)