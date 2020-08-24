import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable, Function, grad
import torch.nn.functional as F
import math
import torchvision

def dice_loss(z, y):
  intersect = torch.sum(z * y)
  y_sum = torch.sum(y)
  z_sum = torch.sum(z)
  loss = 1 - (2 * intersect / (z_sum + y_sum))
  return loss

def dice(x, y):
  assert(x.shape == y.shape)
  intersect = (x * y).sum()
  y_sum = y.sum()
  x_sum = x.sum()
  return 2 * intersect / (x_sum + y_sum)