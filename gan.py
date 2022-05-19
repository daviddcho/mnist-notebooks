#!/usr/bin/env python3
import os
import gzip
import numpy as np 
import torch
import torch.nn as nn
from torch import optim, tensor
from tqdm import trange
import matplotlib.pyplot as plt

# load data
parse = lambda file: np.frombuffer(gzip.open(file).read(), dtype=np.uint8).copy()
X_train = parse("data/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
Y_train = parse("data/train-labels-idx1-ubyte.gz")[8:]

class Generator(nn.Module):
  def __init__(self):
    super(Generator, self).__init__()
    self.layers = nn.Sequential(
      nn.Linear(128, 256),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(256, 512),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(512, 1024),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(1024, 784),
      nn.Tanh()
    )
  
  def forward(self, x):
    return self.layers(x)

class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()
    self.layers = nn.Sequential( 
      nn.Linear(784, 1024),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(1024, 512),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(512, 256),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(256, 1),
      #nn.LogSoftmax(dim=1)
      nn.Sigmoid()
    )

  def forward(self, x):
    return self.layers(x).reshape(-1)

BS = 512 
k = 1 
epochs = 300
n_steps = int(X_train.shape[0]/BS)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device: ", device)

generator = Generator().to(device)
discriminator = Discriminator().to(device)

optim_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optim_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
ds_noise = tensor(np.random.randn(64, 128).astype(np.float32), requires_grad=False).to(device)

def generator_batch():
  samp = np.random.randint(0, X_train.shape[0], size=(BS))
  image = X_train[samp].reshape(-1, 28*28).astype(np.float32)/255.
  image = (image - 0.5)/0.5
  return tensor(image).to(device)

def real_label(bs):
  return torch.full((bs,), 1).float().to(device)

def fake_label(bs):
  return torch.full((bs,), 0).float().to(device)

loss_function = nn.BCELoss()
def train_discriminator(optimizer, real_data, fake_data):
  real_labels = real_label(BS)
  fake_labels = fake_label(BS)

  optimizer.zero_grad()

  real_out = discriminator(real_data)
  real_loss = loss_function(real_out, real_labels)

  fake_out = discriminator(fake_data)
  fake_loss = loss_function(fake_out, fake_labels)

  real_loss.backward() 
  fake_loss.backward()
  optimizer.step()
  return real_loss.item() + fake_loss.item()

def train_generator(optimizer, fake_data):
  y = real_label(BS) 
  optimizer.zero_grad()
  out = discriminator(fake_data)
  loss = loss_function(out, y)
  loss.backward()
  optimizer.step()
  return loss.item()

output_dir = "out"
if not os.path.exists(output_dir):
  os.mkdir(output_dir)

losses_g, losses_d = [], []
for epoch in (t := trange(epochs)):
  loss_g = 0
  loss_d = 0
  for i in range(n_steps):
    # train discriminator
    real_data = generator_batch()
    noise = tensor(np.random.rand(BS, 128)).float().to(device)
    fake_data = generator(noise).detach()
    loss_d_step = train_discriminator(optim_d, real_data, fake_data)
    loss_d += loss_d_step

    # train generator
    noise = tensor(np.random.rand(BS, 128)).float().to(device)
    fake_data = generator(noise).detach()
    loss_g_step = train_generator(optim_g, fake_data)
    loss_g += loss_g_step
  
  fake_images = generator(ds_noise).detach()
  fake_images = ((fake_images.reshape(-1, 28, 28)+1)/2).cpu().numpy()
  fake_images = np.concatenate(fake_images.reshape((8, 8*28, 28)), axis=1)
  plt.figure(figsize=(8,8))
  plt.imshow(fake_images)
  plt.savefig(f"out/images_{epoch}.jpg")
  plt.close()

  epoch_loss_g = loss_g / n_steps
  epoch_loss_d = loss_d / n_steps
  losses_g.append(epoch_loss_g)
  losses_d.append(epoch_loss_d)
  t.set_description("epoch loss_g %.2f loss_d %.2f" % (epoch_loss_g, epoch_loss_d))

plt.plot(epoch_loss_g)
plt.plot(epoch_loss_d)
plt.savefig("loss.png")
plt.close()
