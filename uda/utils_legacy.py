import h5py
import torch
from torch.nn import Parameter
import torch.nn.functional as F
import numpy as np
import copy
import re
import math



def load_data_test(imglabelpath, dataset, length_ratio=-1, is_sr=False, convert_msd=True, min_length=-1):
	data = h5py.File(imglabelpath, 'r')
	image = data['raw']
	label = data['label']

	# change from dataset to array
	image = np.array(image).astype(np.float32)
	label = np.array(label).astype(dtype=np.uint8)

	## do the zero mean and unit variance
	# only caculate the mean and variance values of the positive
	if is_sr:
		image -= image.min()
		image /= image.max()
	else:
		mean_val = np.mean(image[image > 0])
		std_val = np.std(image[image > 0])

		image = (image - mean_val) / std_val

	if length_ratio > 0:
		expected_length = int(length_ratio * image.shape[0])
		if expected_length < min_length:
			expected_length = min_length
		expected_size = (expected_length, ) + image.shape[1:]
		image = np.reshape(image, (1, 1,) + image.shape).astype(np.float32)
		image = F.interpolate(torch.from_numpy(image), size=expected_size, mode='trilinear', align_corners=False).numpy()[0, 0, :, :, :]

	if 'msd' in dataset or 'nih' in dataset:
		label[label > 1] = 0

		if 'liver' in dataset:
			target_class = 5
		elif 'pancreas' in dataset:
			target_class = 6
		elif 'spleen' in dataset:
			target_class = 7
		else:
			print('dataset error')
			exit(-1)
		if convert_msd:
			label[label == 1] = target_class

	return image, label


def construct_PE(image, weight):
	# image is (H, W, L) shape torch tensor
	# return (1, 1, H, W, L)-shaped image if weight <= 0
	assert image.dim() == 3
	if weight <= 0:
		return image.unsqueeze(0).unsqueeze(0)

	device = image.device
	I1 = np.arange(image.shape[-3]).astype(np.float) / (image.shape[-3] - 1)
	I1 = I1[:, np.newaxis, np.newaxis]
	I1 = np.tile(I1, (1, image.shape[-2], image.shape[-1]))
	I2 = np.arange(image.shape[-2]).astype(np.float) / (image.shape[-2] - 1)
	I2 = I2[np.newaxis, :, np.newaxis]
	I2 = np.tile(I2, (image.shape[-3], 1, image.shape[-1]))
	I3 = np.arange(image.shape[-1]).astype(np.float) / (image.shape[-1] - 1)
	I3 = I3[np.newaxis, np.newaxis, :]
	I3 = np.tile(I3, (image.shape[-3], image.shape[-2], 1))

	position_encoding = np.stack([I1, I2, I3]) * weight         # 4, H, W, L
	position_encoding = torch.from_numpy(position_encoding).unsqueeze(0).to(device)

	return position_encoding


def color_map(N=256, normalized=False):
	def bitget(byteval, idx):
		return ((byteval & (1 << idx)) != 0)

	dtype = 'float32' if normalized else 'uint8'
	cmap = np.zeros((N, 3), dtype=dtype)
	for i in range(N):
		r = g = b = 0
		c = i
		for j in range(8):
		r = r | (bitget(c, 0) << 7 - j)
		g = g | (bitget(c, 1) << 7 - j)
		b = b | (bitget(c, 2) << 7 - j)
		c = c >> 3

		cmap[i] = np.array([r, g, b])

	cmap = cmap / 255 if normalized else cmap
	return cmap


def adjust_opt(optAlg, optimizer, init_lr, iter_num, max_iterations, power=0.9):
	if optAlg == 'sgd':
	  lr = init_lr * math.pow(1.0 - (iter_num / max_iterations), power)

	  for param_group in optimizer.param_groups:
		param_group['lr'] = lr


def visualize(im, vote_map, label, n_class=9, ratio=1.0):
	im -= im.min()
	im = (im / im.max() * 255).astype(np.uint8)
	cmap = color_map()
	im = im[..., np.newaxis]
	im = im.repeat(3, axis=-1)
	pre_vis = copy.deepcopy(im)

	for c_idx in range(1, n_class):
		im[..., 0][label == c_idx] = cmap[c_idx, 0] * ratio + im[..., 0][label == c_idx] * (1. - ratio)
		im[..., 1][label == c_idx] = cmap[c_idx, 1] * ratio + im[..., 1][label == c_idx] * (1. - ratio)
		im[..., 2][label == c_idx] = cmap[c_idx, 2] * ratio + im[..., 2][label == c_idx] * (1. - ratio)

		pre_vis[..., 0][vote_map == c_idx] = cmap[c_idx, 0] * ratio + pre_vis[..., 0][vote_map == c_idx] * (1. - ratio)
		pre_vis[..., 1][vote_map == c_idx] = cmap[c_idx, 1] * ratio + pre_vis[..., 1][vote_map == c_idx] * (1. - ratio)
		pre_vis[..., 2][vote_map == c_idx] = cmap[c_idx, 2] * ratio + pre_vis[..., 2][vote_map == c_idx] * (1. - ratio)

	vis = np.concatenate((im, pre_vis), axis=2)
	return vis


def vis_one_image(im, label, n_class=9, ratio=0.8):
	im = im.astype(np.float32)
	im -= im.min()
	im = (im / im.max() * 255).astype(np.uint8)
	cmap = color_map()
	im = im[..., np.newaxis]
	im = im.repeat(3, axis=-1)

	for c_idx in range(1, n_class):
		color_idx = c_idx
		if c_idx == 8:
			color_idx = 11
		if c_idx == 6:
			cmap[c_idx, 1] = 255
			cmap[c_idx, 2] = 255
		im[..., 0][label == c_idx] = cmap[color_idx, 0] * ratio + im[..., 0][label == c_idx] * (1. - ratio)
		im[..., 1][label == c_idx] = cmap[color_idx, 1] * ratio + im[..., 1][label == c_idx] * (1. - ratio)
		im[..., 2][label == c_idx] = cmap[color_idx, 2] * ratio + im[..., 2][label == c_idx] * (1. - ratio)

	return im


def vis_sr_images(im, output_map):
	assert im.min() >= 0 and output_map.min() >= 0
	assert im.max() <= 1 and output_map.max() <= 1

	im = (im * 255.).astype(np.uint8)[..., np.newaxis]
	output_map = (output_map * 255.).astype(np.uint8)[..., np.newaxis]
	im = im.repeat(3, axis=-1)
	output_map = output_map.repeat(3, axis=-1)
	vis = np.concatenate((im, output_map), axis=2)
	return vis


def cal_histogram(im, label, cls):
	assert im.min() >= 0 and im.max() <= 1
	im = (im * 255.).astype(np.uint8)

	im = im[label == cls].ravel()
	hist = np.bincount(im, minlength=256)
	return hist


def load_state_dict(net, state_dict, remove='', add=''):
	own_state = net.state_dict()
	for param in own_state.items():
		name = add + param[0].replace(remove, '')
		if name not in state_dict:
		print('{} not in pretrained model'.format(param[0]))
	for name, param in state_dict.items():
		if remove + name.replace(add, '') not in own_state:
		print('skipping {}'.format(name))
		continue
		if isinstance(param, Parameter):
		# backwards compatibility for serialized parameters
		param = param.data
		if param.shape == own_state[remove + name.replace(add, '')].shape:
		own_state[remove + name.replace(add, '')].copy_(param)
		else:
		print('skipping {} because of shape inconsistency'.format(name))


def dice(x, y, eps=1e-7):
	intersect = np.sum(np.sum(np.sum(x * y)))
	y_sum = np.sum(np.sum(np.sum(y)))
	x_sum = np.sum(np.sum(np.sum(x)))
	return 2 * intersect / (x_sum + y_sum + eps)


def binary_dice_loss(input, target):
	smooth = 1.

	# apply softmax to input
	input = F.softmax(input, dim=1)
	input = input[:, 1]

	iflat = input.contiguous().view(-1)
	tflat = target.contiguous().view(-1)
	intersection = (iflat * tflat).sum()

	loss = 1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))
	return loss


def sort_nicely(l):
	""" Sort the given list in the way that humans expect.
	"""
	convert = lambda text: int(text) if text.isdigit() else text
	alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
	l.sort( key=alphanum_key )
	return l

