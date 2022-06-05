#!/usr/bin/env python3
import torch
import torch.nn as nn
#from dcgan import PrintLayer

class VAE(nn.Module):
  def __init__(self, zdim=4):
    super(VAE, self).__init__()
    self.conv = nn.Sequential(
      nn.Conv2d(1, 64, 2, 2),
      nn.BatchNorm2d(64), 
      nn.ReLU(),
      nn.Conv2d(64, 128, 2, 2),
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.Conv2d(128, 256, 4, 2),
      nn.BatchNorm2d(256),
      nn.ReLU(),
    )
    self.fc1 = nn.Linear(1024, zdim)
    self.fc2 = nn.Linear(1024, zdim)

    self.fc3 = nn.Sequential(
      nn.Linear(zdim, 1024),
      nn.BatchNorm1d(1024),
      nn.ReLU(),
    )
    self.conv_transpose = nn.Sequential(
      nn.ConvTranspose2d(1024, 128, 5, 2), 
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
    return self.fc1(x), self.fc2(x)

  def reparameterize(self, mu, logvar):
    std = torch.exp(logvar*0.5)
    eps = torch.randn_like(std)
    return mu + eps*std
 
  def decoder(self, x):
    x = self.fc3(x)
    x = x.reshape(x.shape[0], -1, 1, 1)
    x = self.conv_transpose(x)
    return x

  def forward(self, x):
    mu, logvar = self.encoder(x)
    z = self.reparameterize(mu, logvar)
    return self.decoder(z), mu, logvar

if __name__ == "__main__":
  vae = VAE()
  x = torch.rand(4, 1, 28, 28)
  out = vae(x)
