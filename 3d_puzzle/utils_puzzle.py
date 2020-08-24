import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable, Function, grad
import torch.nn.functional as F
import math
import torchvision

def unary_loss(output_list, label_list):
  loss_list = []
  puzzle_num = output_list[0].shape[2]
  loss = nn.NLLLoss()

  for i in range(len(output_list)):
      output = output_list[i].view(-1, puzzle_num)
      label = label_list[i].view(-1)
      loss_list.append(loss(output, label))
  return torch.mean(torch.stack(loss_list))

def binary_loss(b_stack, b_label_batch):
  b_label_batch = b_label_batch.view(-1)
  loss = nn.NLLLoss()

  return loss(b_stack, b_label_batch)
