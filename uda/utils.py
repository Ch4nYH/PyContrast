import os
import math
import torch
import shutil
import tensorboardX

import numpy as np
import torch.nn.functional as F

def cross_entropy_3d(x, y, w=None):
	assert len(x.shape) == len(y.shape)
	n, c, _, _, _ = x.size()
	x_t = torch.transpose(torch.transpose(torch.transpose(x, 1, 2), 2, 3), 3, 4).contiguous().view(-1, c)
	y_t = torch.transpose(torch.transpose(torch.transpose(y, 1, 2), 2, 3), 3, 4).contiguous().view(-1).long()
	loss = F.cross_entropy(x_t, y_t, weight=w)
	return loss
def dice(x, y, eps=1e-7):
  intersect = np.sum(np.sum(np.sum(x * y)))
  y_sum = np.sum(np.sum(np.sum(y)))
  x_sum = np.sum(np.sum(np.sum(x)))
  return 2 * intersect / (x_sum + y_sum + eps)

def adjust_learning_rate(args, optimizer, epoch):
	for param_group in optimizer.param_groups():
		param_group['lr'] = args.lr * math.pow(1.0 - (epoch / args.epochs), 0.9)

class Logger(object):

	def __init__(self, path):
		self.global_step = 0
		self.logger = tensorboardX.SummaryWriter(os.path.join(path, "log"))

	def log(self, name, value):
		self.logger.add_scalar(name, value, self.global_step)

	def step(self):
		self.global_step += 1

	def close(self):
		self.logger.close()

class Saver(object):

	def __init__(self, path, save_interval = 10):
		self.path = path
		self.best_dice = 0
		self.save_interval = save_interval
	def save(self, epoch, states, test_dice):
		if epoch % self.save_interval == 0:
			torch.save(states, os.path.join(self.path, 'model', 'checkpoint_{}.pth.tar'.format(epoch)))
			if test_dice > self.best_dice:
				torch.save(states, os.path.join(self.path, 'model', 'checkpoint_best.pth.tar'))
				self.best_dice = test_dice