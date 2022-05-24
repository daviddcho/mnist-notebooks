#!/usr/bin/env python3
import os
import gzip
import numpy as np 
import torch
import torch.nn as nn
from torch import optim, tensor
import torch.nn.functional as F
from tqdm import trange
import matplotlib.pyplot as plt

# load data
parse = lambda file: np.frombuffer(gzip.open(file).read(), dtype=np.uint8).copy()
X_train = parse("data/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
Y_train = parse("data/train-labels-idx1-ubyte.gz")[8:]

from model import Generator, Discriminator

if __name__ == "__main__":
  BS = 128
  k = 1 
  epochs = 50 #300
  n_steps = int(X_train.shape[0]/BS)
  z_dim = 100

  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  print("device: ", device)

  output_dir = "out"
  os.makedirs(output_dir, exist_ok=True)

  generator = Generator().to(device)
  discriminator = Discriminator().to(device)

  optim_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
  optim_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
  #ds_noise = tensor(np.random.randn(64, z_dim).astype(np.float32), requires_grad=False).to(device)
  ds_noise = tensor(np.random.randn(64, z_dim, 1, 1).astype(np.float32), requires_grad=False).to(device)

  def generator_batch():
    samp = np.random.randint(0, X_train.shape[0], size=(BS))
    #image = X_train[samp].reshape(-1, 28*28).astype(np.float32)/255.
    image = X_train[samp].astype(np.float32)/255.
    image = (image - 0.5)/0.5
    image = image.reshape(-1, 1, 28, 28)
    return tensor(image).to(device)

  def train_discriminator(optimizer, real_data, fake_data):
    optimizer.zero_grad()

    out = discriminator(real_data)
    y = torch.ones(BS, dtype=torch.float32).to(device)
    real_loss = F.binary_cross_entropy(out, y)

    out = discriminator(fake_data)
    y = torch.zeros(BS, dtype=torch.float32).to(device)
    fake_loss = F.binary_cross_entropy(out, y)

    real_loss.backward() 
    fake_loss.backward()
    optimizer.step()
    return real_loss.item() + fake_loss.item()

  def train_generator(optimizer, fake_data):
    y = torch.ones(BS, dtype=torch.float32).to(device)
    optimizer.zero_grad()
    out = discriminator(fake_data)
    loss = F.binary_cross_entropy(out, y)
    loss.backward()
    optimizer.step()
    return loss.item()

  losses_g, losses_d = [], []
  for epoch in (t := trange(epochs)):
    loss_g = 0
    loss_d = 0
    for i in trange(n_steps):
      for step in range(k):
        # train discriminator
        real_data = generator_batch()
        noise = tensor(np.random.rand(BS, z_dim, 1, 1)).float().to(device)

        fake_data = generator(noise).detach()
        loss_d_step = train_discriminator(optim_d, real_data, fake_data)
        loss_d += loss_d_step

      # train generator
      noise = tensor(np.random.rand(BS, z_dim, 1, 1)).float().to(device)
      fake_data = generator(noise).detach()
      loss_g_step = train_generator(optim_g, fake_data)
      loss_g += loss_g_step

    losses_g.append(loss_g)
    losses_d.append(loss_d)

    # generate from fixed noise  
    fake_images = generator(ds_noise).detach()
    #fake_images = ((fake_images.reshape(-1, 28, 28)+1)/2).cpu().numpy()
    fake_images = fake_images.reshape(-1, 28, 28).cpu().numpy()
    fake_images = np.concatenate(fake_images.reshape((8, 8*28, 28)), axis=1)
    plt.figure(figsize=(8,8))
    plt.imshow(fake_images)
    plt.savefig(f"out/images_{epoch}.jpg")
    #plt.close()

    epoch_loss_g = loss_g / n_steps
    epoch_loss_d = loss_d / n_steps
    t.set_description("epoch loss_g %.2f loss_d %.2f" % (epoch_loss_g, epoch_loss_d))

  """
  plt.plot(epoch_loss_g)
  plt.plot(epoch_loss_d)
  plt.savefig("loss.png")
  plt.close()
  """
