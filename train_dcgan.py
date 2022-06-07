#!/usr/bin/env python3
import os
import gzip
import time
import numpy as np 
import torch
import torch.nn as nn
from torch import optim, tensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import trange
from models.dcgan import Generator, Discriminator

def load_data():
  parse = lambda file: np.frombuffer(gzip.open(file).read(), dtype=np.uint8).copy()
  X_train = parse("data/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
  X_test = parse("data/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
  X_train = np.concatenate((X_train, X_test), axis=0)
  return X_train
X_train = load_data()

batch_size = 128
n_epochs = 30
n_batches = int(X_train.shape[0]/batch_size)
z_dim = 100

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device: ", device)

generator = Generator().to(device)
discriminator = Discriminator().to(device)

import apex
optim_g = apex.optimizers.FusedAdam(generator.parameters(), lr=0.0002)
optim_d = apex.optimizers.FusedAdam(discriminator.parameters(), lr=0.0002)
#optim_g = optim.Adam(generator.parameters(), lr=0.0002)
#optim_d = optim.Adam(discriminator.parameters(), lr=0.0002)
fixed_noise = torch.randn(64, 100, 1, 1, device=device)

def generate_batch():
  samp = np.random.randint(0, X_train.shape[0], size=(batch_size))
  x = X_train[samp].astype(np.float32)/255.
  x = tensor(x.reshape(-1, 1, 28, 28)).to(device)
  return x

def train_discriminator(real_data, fake_data):
  optim_d.zero_grad()

  out = discriminator(real_data)
  y = torch.ones(batch_size).float().to(device)
  real_loss = F.binary_cross_entropy(out, y)
  real_loss.backward() 

  out = discriminator(fake_data.detach())
  y = torch.zeros(batch_size).float().to(device)
  fake_loss = F.binary_cross_entropy(out, y)
  fake_loss.backward()

  optim_d.step()
  return real_loss + fake_loss

def train_generator(fake_data):
  y = torch.ones(batch_size).float().to(device)
  optim_g.zero_grad()
  out = discriminator(fake_data)
  loss = F.binary_cross_entropy(out, y)
  loss.backward()
  optim_g.step()
  return loss

out_dir = "out"
os.makedirs(out_dir, exist_ok=True)
def save_fake_images(generator, fixed_noise, epoch, timestamp):
  fake_images = generator(fixed_noise)
  fake_images = fake_images.reshape(-1, 28, 28).cpu().detach().numpy()
  fake_images = np.concatenate(fake_images.reshape((8, 28*8, 28)), axis=1)
  plt.figure(figsize=(8,8))
  plt.imshow(fake_images)
  plt.savefig(f"{out_dir}/images_{timestamp}_{epoch}.png")
  plt.close()

timestamp = int(time.time())
losses_g, losses_d = [], []
for epoch in trange(n_epochs):
  for i in (t := trange(n_batches)):
    # train discriminator
    real_data = generate_batch()
    noise = torch.randn(batch_size, z_dim, 1, 1).to(device)
    fake_data = generator(noise)
    loss_d = train_discriminator(real_data, fake_data)

    # train generator
    loss_g = train_generator(fake_data)

    t.set_description("loss_g %.4f loss_d %.4f" % (loss_g.item(), loss_d.item()))
    losses_g.append(loss_g.item())
    losses_d.append(loss_d.item())
  save_fake_images(generator, fixed_noise, epoch, timestamp)
  
"""
plt.plot(losses_g, label="generator loss")
plt.plot(losses_d, label="discriminator loss")
plt.legend()
plt.savefig("loss.png")
plt.close()
"""
