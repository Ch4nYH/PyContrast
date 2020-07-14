
import numpy as np

class RandomCrop(object):
  """
  Crop randomly the image in a sample
  Args:
	output_size (int): Desired output size
  """

  	def __init__(self, output_size, sample_num=1, pad=32, is_binary=False):
		self.output_size = output_size
		self.sample_num = sample_num
		self.pad = pad
		self.is_binary = is_binary

  	def __call__(self, sample):
		image, label = sample['image'], sample['label']
		assert image.shape == label.shape
		if self.is_binary and label.max() > 1:
			label[label > 1] = 0
		if any(np.array(image.shape) <= self.output_size):
			pad_x = max(0, self.output_size - image.shape[0] + 1)
			pad_y = max(0, self.output_size - image.shape[1] + 1)
			pad_z = max(0, self.output_size - image.shape[2] + 1)
			image = np.pad(image, ((0, pad_x), (0, pad_y), (0, pad_z)), 'mean')
			label = np.pad(label, ((0, pad_x), (0, pad_y), (0, pad_z)), 'constant')

		if self.pad < 0:
			bbox = [[0, label.shape[0]], [0, label.shape[1]], [0, label.shape[2]]]
		else:
			tempL = np.nonzero(label)
			bbox = [[max(0, np.min(tempL[0]) - self.pad), min(label.shape[0], np.max(tempL[0]) + 1 + self.pad)],
					[max(0, np.min(tempL[1]) - self.pad), min(label.shape[1], np.max(tempL[1]) + 1 + self.pad)],
					[max(0, np.min(tempL[2]) - self.pad), min(label.shape[2], np.max(tempL[2]) + 1 + self.pad)]]

		# crop random sample on whole image
		output_image = np.zeros((self.sample_num, self.output_size, self.output_size, self.output_size))
		output_label = np.zeros((self.sample_num, self.output_size, self.output_size, self.output_size))
		for i in range(self.sample_num):
			if bbox[0][1] - self.output_size <= bbox[0][0]:
				print(bbox[0])
			w1 = np.random.randint(bbox[0][0], bbox[0][1] - self.output_size + 1)
			h1 = np.random.randint(bbox[1][0], bbox[1][1] - self.output_size + 1)
			d1 = np.random.randint(bbox[2][0], bbox[2][1] - self.output_size + 1)
			output_image[i] = image[w1:w1+self.output_size, h1:h1+self.output_size, d1:d1+self.output_size]
			output_label[i] = label[w1:w1+self.output_size, h1:h1+self.output_size, d1:d1+self.output_size]
		return {'image': output_image, 'label': output_label, 'ori_image': copy.deepcopy(image)}


class RandomTranspose(object):
	'''
	Randomly transpose axis
	'''
	def __call__(self, sample):
		image, label = sample['image'], sample['label']
		assert image.shape == label.shape

		if image.ndim == 3:
			image, label = self._trans(image, label)
		elif image.ndim == 4:
			for i in range(len(image)):
				image[i], label[i] = self._trans(image[i], label[i])
		else:
			print('dim error')
			exit(-1)

		sample['image'] = image
		sample['label'] = label
		for key in sample:
			if key not in ['image', 'label']:
				sample[key] = copy.deepcopy(sample[key])
		return sample

	def _trans(self, image, label):
		pp = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
		degree = random.choice([0, 1, 2])
		image, label = np.transpose(image, pp[degree]), np.transpose(label, pp[degree])
		return image, label

class RandomRotate(object):
	'''
	Randomly rotate the image
	'''
	def __call__(self, sample):
		image, label = sample['image'], sample['label']
		if image.ndim == 3:
			image, label = self._rotate(image, label)
		elif image.ndim == 4:
			for i in range(len(image)):
				image[i], label[i] = self._rotate(image[i], label[i])
		else:
			print('dim error')
			exit(-1)

		sample['image'] = image
		sample['label'] = label
		for key in sample:
			if key not in ['image', 'label']:
				sample[key] = copy.deepcopy(sample[key])
		return sample

	def _rotate(self, x, y):
		degree = random.choice([0, 1, 2, 3])
		x, y = np.rot90(x, degree, (0, 1)), np.rot90(y, degree, (0, 1))
		return x, y


class ToTensor(object):
  """Convert ndarrays in sample to Tensors."""

  def __call__(self, sample):
	image, label = sample['image'], sample['label']
	if image.ndim >= 5:                                                     # already has channel dim
		image = torch.from_numpy(image.astype(np.float32))
	else:
		image = torch.from_numpy(image.astype(np.float32)).unsqueeze(1)     # Crop_num, 1, h, w, l
	label = torch.from_numpy(label.astype(np.float32)).unsqueeze(1)
	return {'image': image, 'label': label}


def build_transforms(args):

	train_transforms = [RandomCrop(64, 8, pad=-1, is_binary=True),
					RandomTranspose(),
					RandomRotate(),
					ToTensor()]
	test_transforms = [RandomCrop(64, 8, pad=48, is_binary=True),
						ToTensor()]
	return train_transforms, test_transforms

def get_jigsaw_transforms(args):
	'''
	jigsaw_transform = transforms.Compose([
		transforms.RandomResizedCrop(255, scale=(0.6, 1)),
		transforms.RandomHorizontalFlip(),
		JigsawCrop(),
		StackTransform(transforms.Compose([
			color_transfer,
			transforms.ToTensor(),
			normalize,
		]))
	])
	'''
	return None
