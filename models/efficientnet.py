#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MBConvBlock(nn.Module):
  """Mobile Inverted Residual Bottleneck Block"""
  def __init__(self, kernel_size, strides, expand_ratio, input_filters, output_filters, se_ratio, has_se):
    super(MBConvBlock, self).__init__()
    oup = expand_ratio * input_filters
    if expand_ratio != 1:
      self._expand_conv = nn.Conv2d(in_channels=input_filters, out_channels=oup, kernel_size=1, bias=False)
      self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=0.1, eps=1e-5)
    else:
      self._expand_conv = None

    self.strides = strides
    if strides == (2,2):
      self.pad = [(kernel_size-1)//2-1, (kernel_size-1)//2]*2
    else:
      self.pad = [(kernel_size-1)//2]*4

    k = kernel_size
    s = self.strides
    print(k, s)
    self._depthwise_conv = nn.Conv2d(in_channels=oup, out_channels=oup, groups=oup,
                                     kernel_size=k, stride=s, bias=False)
    self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=0.1, eps=1e-5)

    self.has_se = has_se
    if self.has_se:
      num_squeezed_channels = max(1, int(input_filters * se_ratio))
      self._se_reduce = nn.Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
      self._se_expand = nn.Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

    self._project_conv = nn.Conv2d(in_channels=oup, out_channels=output_filters, kernel_size=1, bias=False)
    self._bn2 = nn.BatchNorm2d(num_features=output_filters, momentum=0.1, eps=1e-5)
    self._swish = nn.SiLU()

  def pad2d(self, x, padding):
    return x[:, :, -padding[2]:x.shape[2]+padding[3], -padding[0]:x.shape[3]+padding[1]]

  def forward(self, inputs):
    # Expansion and Depthwise Convolution
    x = inputs
    if self._expand_conv:
      x = self._expand_conv(x)
      x = self._bn0(x)
      x = self._swish(x)

    print(x.shape)
    #x = self.pad2d(x, self.pad)
    print(x.shape)
    x = self._depthwise_conv(x)
    x = self._bn1(x)
    x = self._swish(x)

    # Squeeze and Excitation
    if self.has_se:
      x_squeezed = F.adaptive_avg_pool2d(x, 1)
      x_squeezed = self._se_reduce(x_squeezed)
      x_squeezed = self._swish(x_squeezed)
      x_squeezed = self._se_expand(x_squeezed)
      x = torch.sigmoid(x_squeezed) * x

    # Pointwise Convolution
    x = self._project_conv(x)
    x = self._bn2(x)

    # Skip connection
    if x.shape == inputs.shape:
      x = x + inputs
    return x

class EfficientNet(nn.Module):
  def __init__(self, number=0, classes=1000, has_se=True):
    super(EfficientNet, self).__init__()
    self.number = number 
    global_params = [
      # width, depth 
      (1.0, 1.0), # b0 
      (1.0, 1.1), # b1 
      (1.1, 1.2), # b2
      (1.2, 1.4), # b3
      (1.4, 1.8), # b4
      (1.6, 2.2), # b5
      (1.8, 2.6), # b6
      (2.0, 3.1), # b7
      (2.2, 3.6), # b8
      (4.3, 5.3), # l2
    ][number]
    
    def round_filters(filters):
      """Round number of filters based on depth multiplier"""
      multiplier = global_params[0]
      divisor = 8
      filters *= multiplier 
      new_filters = max(divisor, int(filters + divisor/2) // divisor*divisor)
      if new_filters < 0.9 * filters: # prevent rounding by more than 10%
        new_filters += divisor
      return int(new_filters)

    def round_repeats(repeats):
      return int(math.ceil(global_params[1] * repeats))
    
    # Stem
    in_channels = 3 # rgb
    out_channels = round_filters(32)
    self._conv_stem = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
    self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=0.1, eps=1e-5)
    
    # num_repeats, kernel_size, strides, expand_ratio, input_filters, output_filters, se_ratio
    block_args = [
      [1, 3, (1,1), 1, 32, 16, 0.25],
      [2, 3, (2,2), 6, 16, 24, 0.25],
      [2, 5, (2,2), 6, 24, 40, 0.25],
      [3, 3, (2,2), 6, 40, 80, 0.25],
      [3, 5, (1,1), 6, 80, 112, 0.25],
      [4, 5, (2,2), 6, 112, 192, 0.25],
      [1, 3, (1,1), 6, 192, 320, 0.25],
    ]
    self._blocks = []
    # Build blocks
    for b in block_args:
      args = b[1:]
      args[3] = round_filters(args[3])
      args[4] = round_filters(args[4])
      for n in range(round_repeats(b[0])):
        self._blocks.append(MBConvBlock(*args, has_se=has_se))
        args[3] = args[4]
        args[1] = (1,1)
    
    # Head
    in_channels = round_filters(320)
    out_channels = round_filters(1280)
    self._conv_head = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
    self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=0.1, eps=1e-5)
    
    # Final linear layer 
    self._avg_pooling = nn.AdaptiveAvgPool2d(1)
    self._dropout = nn.Dropout(0.2)
    self._fc = nn.Linear(out_channels, classes)
    self._swish = nn.SiLU()
    
  def forward(self, x):
    print(x.shape)
    x = self._swish(self._bn0(self._conv_stem(x)))
    for block in self._blocks:
      x = block(x)
    x = self._swish(self._bn1(self._conv_head(x)))
    x = self._avg_pooling(x)
    x = self._dropout(x)
    x = self._fc(x)
    return x

if __name__ == "__main__":
  model = EfficientNet(0, classes=10, has_se=False)
  x = torch.randn(4, 3, 32, 32)
  out = model(x)
  print(out)
  """
  loss_fn = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters())
  for i in (t := trange(5)):
    samp = np.random.randint(0, X_train.shape[0], size=16)
    X = torch.tensor(X_train[samp]).float()
    out = model(X)
  """
