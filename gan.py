#!/usr/bin/env python3
import gzip
import numpy as np 
import torch
import torch.nn as nn
from tqdm import trange

# load data
parse = lambda file: np.frombuffer(gzip.open(file).read(), dtype=np.uint8).copy()
X_train = parse("data/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
Y_train = parse("data/train-labels-idx1-ubyte.gz")[8:]
X_test = parse("data/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
Y_test = parse("data/t10k-labels-idx1-ubyte.gz")[8:]

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
      nn.Linear(256, 2),
      nn.LogSoftmax(dim=1)
    )

  def forward(self, x):
    return self.layers(x)

BS = 512 
k = 1 
epochs = 300
n_steps = int(X_train.shape[0]/BS)

generator = Generator()
discriminator = Discriminator()

optim_g = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optim_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
ds_noise = torch.tensor(np.random.randn(64, 128).astype(np.float32), requires_grad=False)

def generator_batch():
  samp = np.random.randint(0, X_train.shape[0], size=(BS))
  image = X_train[samp].reshape(-1, 28*28).astype(np.float32)/255.
  image = (image - 0.5)/0.5
  return torch.tensor(image)

def train_discriminator(optimizer, real_data, fake_data):
  pass

def train_generator(optimizer, fake_data):
  pass

for epoch in (t := trange(epochs)):
  loss_g = 0
  loss_d = 0
  for i in range(n_steps):
    # train discriminator
    real_data = generator_batch()
    noise = torch.tensor(np.random.rand(BS, 128)).float()
    fake_data = generator(noise).detach()
    loss_d_step = train_discriminator(optim_d, real_fake, fake_data)
    loss_d += loss_d_step

    # train generator
    noise = torch.tensor(np.random.rand(BS, 128)).float()
    fake_data = generator(noise).detach()
    loss_g_step = train_generator(optim_g, fake_data)
    loss_g += loss_g_step
  
  fake_images = generator(ds_noise).detach()
  fake_images = ((fake_images.reshape(-1, 28, 28)+1)/2).numpy()
  fake_images = np.concatenate(fake_images.reshape((8, 8*28, 28)), axis=1)
  plt.figure(figsize=(8,8))
  plt.imshow(fake_images)
  plt.savefig(f"out/images_{epoch}.jpg")

  epoch_loss_g = loss_g / n_steps
  epoch_loss_d = loss_d / n_steps
  t.set_description("epoch loss_g %.2f loss_d %.2f" % (epoch_loss_g, epoch_loss_d))

