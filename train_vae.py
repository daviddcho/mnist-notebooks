#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
import matplotlib.pyplot as plt
import numpy as np
import gzip
import time

parse = lambda file: np.frombuffer(gzip.open(file).read(), dtype=np.uint8).copy()
X_train = parse("data/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
X_test = parse("data/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))

batch_size = 128
n_epochs = 100  
n_batches = len(X_train)//batch_size

from models.vae import VAE
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = VAE(zdim=20).to(device)
optimizer = torch.optim.Adam(model.parameters())
def loss_function(x, target, mu, logvar):
  mse_loss = F.mse_loss(x, target, reduction='sum')
  #mse_loss = F.binary_cross_entropy(x, target, reduction='sum')
  kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  return mse_loss + kld_loss

timestamp = int(time.time())
for epoch in trange(n_epochs):
  for i in (t := trange(n_batches)):
    samp = np.random.randint(0, X_train.shape[0], size=(batch_size))
    X = torch.tensor(X_train[samp]).reshape(-1, 1, 28, 28).float().to(device)
    optimizer.zero_grad()
    out, mu, logvar = model(X)
    loss = loss_function(out, X, mu, logvar)
    loss.backward()
    optimizer.step()
    t.set_description(f"loss {loss:.2f}")

  samp = np.random.randint(0, X_test.shape[0], size=(batch_size//2))
  X = torch.tensor(X_test[samp]).reshape(-1, 1, 28, 28).float().to(device)
  images, _, _ = model(X)
  images = images.cpu().detach().numpy().reshape(-1, 28, 28)
  images = np.concatenate(images.reshape(8, 28*8, 28), axis=1)
  plt.imshow(images)
  plt.savefig(f"out/images_{timestamp}_{epoch}")