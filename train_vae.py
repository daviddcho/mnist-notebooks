#!/usr/bin/env python3
import torch
import torch.nn as nn
from tqdm import trange
import matplotlib.pyplot as plt
import numpy as np
import gzip

parse = lambda file: np.frombuffer(gzip.open(file).read(), dtype=np.uint8).copy()
X_train = parse("data/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
X_test = parse("data/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))

batch_size = 128
n_epochs = 100  
n_batches = len(X_train)//batch_size

from models.vae import VAE
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = VAE().to(device)
optimizer = torch.optim.Adam(model.parameters())
loss_function = nn.MSELoss()

for epoch in trange(n_epochs):
  for i in (t := trange(n_batches)):
    samp = np.random.randint(0, X_train.shape[0], size=(batch_size))
    X = torch.tensor(X_train[samp]).reshape(-1, 1, 28, 28).float().to(device)
    optimizer.zero_grad()
    out = model(X)
    loss = loss_function(out, X)
    loss.backward()
    optimizer.step()
    t.set_description(f"loss {loss:.2f}")

  samp = np.random.randint(0, X_test.shape[0], size=(batch_size//2))
  X = torch.tensor(X_test[samp]).reshape(-1, 1, 28, 28).float().to(device)
  out = model(X).cpu().detach().numpy()
  images = out.reshape(-1, 28, 28)
  images = np.concatenate(out.reshape(8, 28*8, 28), axis=1)
  plt.imshow(images)
  plt.savefig(f"out/images_{epoch}")