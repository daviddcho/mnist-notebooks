#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SqueezeExcite(nn.Module):
  def __init__(self, out_chs, r_chs):
    super(SqueezeExcite, self).__init__()
    self.se_reduce = nn.Conv2d(out_chs, r_chs, 1)
    self.se_expand = nn.Conv2d(r_chs, out_chs, 1)
    self.swish = nn.SiLU()
  
  def forward(self, x):
    x_squeezed = F.adaptive_avg_pool2d(x, 1)
    x_squeezed = self.se_reduce(x_squeezed)
    x_squeezed = self.swish(x_squeezed)
    x_squeezed = self.se_expand(x_squeezed)
    x = torch.sigmoid(x_squeezed) * x
    return x

class MBConvBlock(nn.Module):
  """
  Mobile Inverted Residual Bottleneck Block
  """
  def __init__(self, kernel_size, strides, expand_ratio, input_filters, output_filters, se_ratio, has_se):
    super(MBConvBlock, self).__init__()
    out_chs = expand_ratio * input_filters
    if expand_ratio != 1:
      self.expand_conv = nn.Conv2d(input_filters, out_chs, 1, bias=False)
      self.bn0 = nn.BatchNorm2d(out_chs)
    else:
      self.expand_conv = None

    self.padding = self.get_padding(kernel_size, strides)
    self.pad2d = nn.ZeroPad2d(self.padding)
    self.depthwise_conv = nn.Conv2d(out_chs, out_chs, kernel_size, strides, groups=out_chs, bias=False)
    self.bn1 = nn.BatchNorm2d(out_chs)

    # Squeeze and Excitation
    self.has_se = has_se
    if self.has_se:
      num_squeezed_channels = max(1, int(input_filters * se_ratio))
      self.se_conv = SqueezeExcite(out_chs, num_squeezed_channels)
      
    self.project_conv = nn.Conv2d(out_chs, output_filters, 1, bias=False)
    self.bn2 = nn.BatchNorm2d(output_filters)
    self.swish = nn.SiLU()

  def get_padding(self, kernel_size, strides):
    p = max(kernel_size-1, 0)
    return [p//2, p-(p//2), p//2, p-(p//2)] 

  def forward(self, inputs):
    # Expansion and Depthwise Convolution
    x = inputs
    if self.expand_conv:
      x = self.expand_conv(x)
      x = self.bn0(x)
      x = self.swish(x)

    x = self.pad2d(x)
    x = self.depthwise_conv(x)
    x = self.bn1(x)
    x = self.swish(x)

    # Squeeze and Excitation
    if self.has_se:
      x = self.se_conv(x)
      
    # Pointwise Convolution
    x = self.project_conv(x)
    x = self.bn2(x)

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
    
    def round_filters(filters, divisor=8):
      """Round number of filters based on depth multiplier"""
      multiplier = global_params[0]
      filters *= multiplier 
      new_filters = max(divisor, int(filters + divisor/2) // divisor*divisor)
      if new_filters < 0.9 * filters: # prevent rounding by more than 10%
        new_filters += divisor
      return int(new_filters)

    def round_repeats(repeats):
      return int(math.ceil(global_params[1] * repeats))
    
    # Stem
    out_chs = round_filters(32)
    self.conv_stem = nn.Conv2d(3, out_chs, 3, 2, bias=False)
    self.bn0 = nn.BatchNorm2d(out_chs)
    
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
    self.blocks = []
    # Build blocks
    for b in block_args:
      args = b[1:]
      args[3] = round_filters(args[3])
      args[4] = round_filters(args[4])
      for n in range(round_repeats(b[0])):
        self.blocks.append(MBConvBlock(*args, has_se=has_se))
        args[3] = args[4]
        args[1] = (1,1)
    
    # Head
    in_chs = round_filters(320)
    out_chs = round_filters(1280)
    self.conv_head = nn.Conv2d(in_chs, out_chs, kernel_size=1, bias=False)
    self.bn1 = nn.BatchNorm2d(out_chs, momentum=0.1, eps=1e-5)
    
    # Final linear layer 
    self.avg_pooling = nn.AdaptiveAvgPool2d(1)
    self.dropout = nn.Dropout(0.2)
    self.fc = nn.Linear(out_chs, classes)
    self.swish = nn.SiLU()
    
  def forward(self, x):
    x = self.swish(self.bn0(self.conv_stem(x)))
    for block in self.blocks:
      x = block(x)

    x = self.swish(self.bn1(self.conv_head(x)))
    x = self.avg_pooling(x)
    x = self.dropout(x)
    x = x.mean([2, 3])
    x = self.fc(x)
    return x

if __name__ == "__main__":
  torch.manual_seed(1229)
  model = EfficientNet(number=0, classes=10, has_se=True)
  #x = torch.randn(4, 3, 32, 32)
  #torch.save(x, "tensor.pt")
  x = torch.load("tensor.pt")
  out = model(x)
  print(out)