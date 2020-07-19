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
	for param_group in optimizer.param_groups:
		param_group['lr'] = args.lr * math.pow(1.0 - (epoch / args.epochs), 0.9)

def compute_loss_accuracy(logits, target, criterion):
	"""
	Args:
	  logits: a list of logits, each with a contrastive task
	  target: contrastive learning target
	  criterion: typically nn.CrossEntropyLoss
	"""
	losses = [criterion(logit, target) for logit in logits]

	def acc(l, t):
		acc1 = accuracy(l, t)
		return acc1[0]

	accuracies = [acc(logit, target) for logit in logits]

	return losses, accuracies

def accuracy(output, target, topk=(1,)):
	"""Computes the accuracy over the k top predictions for the specified values of k"""
	with torch.no_grad():
		maxk = max(topk)
		batch_size = target.size(0)

		_, pred = output.topk(maxk, 1, True, True)
		pred = pred.t()
		correct = pred.eq(target.view(1, -1).expand_as(pred))

		res = []
		for k in topk:
			correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
			res.append(correct_k.mul_(100.0 / batch_size))
		return res


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

	def __init__(self, path, save_freq = 10):
		self.path = path
		self.best_dice = 0
		self.save_freq = save_freq
	def save(self, epoch, states, test_dice):
		if epoch % self.save_freq == 0:
			torch.save(states, os.path.join(self.path, 'model', 'checkpoint_{}.pth.tar'.format(epoch)))
			if test_dice > self.best_dice:
				torch.save(states, os.path.join(self.path, 'model', 'checkpoint_best.pth.tar'))
				self.best_dice = test_dice

class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

