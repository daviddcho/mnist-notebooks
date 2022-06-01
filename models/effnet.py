#!/usr/bin/env python3
import torch 
import numpy as np
from efficientnet_pytorch import EfficientNet

np.random.seed(1337)
model = EfficientNet.from_name("efficientnet-b0", num_classes=10)
x = torch.load("tensor.pt")
out = model(x)
print(out)
