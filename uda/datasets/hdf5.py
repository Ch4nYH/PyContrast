import os
import numpy as np
import h5py
from torch.utils.data import Dataset
from tqdm import tqdm

class HDF5Dataset(Dataset):
	""" JHU Dataset """

	def __init__(self, list_file, root_dir, transform=None, read_label=True, need_non_zero_label = True, is_binary = False):
		"""
		Args:
		  list_file (string) : List of the image files
		  root_dir (string) : Directory of images
		  transform (callable, optional) : Optional transform to be applied on a sample
		"""
		self.image_list = open(list_file).readlines()
		self.image_list = [os.path.basename(line.strip()) for line in self.image_list]
		self.image_list = [line for line in self.image_list if line.endswith('.h5')]
		
		self.root_dir = root_dir
		self.read_label = read_label
		self.transform = transform
		if need_non_zero_label:
			self.temp = []
			pbar = tqdm(self.image_list)
			for img in pbar:
				img_name = os.path.join(self.root_dir, img)
				data = h5py.File(img_name, 'r')
				label = np.array(data['label'])
				if is_binary and label.max() > 1:
					label[label > 1] = 0
				if len(np.nonzero(label)[0]) * len(np.nonzero(label)[1]) * len(np.nonzero(label)[2]) > 0:
					self.temp.append(img)
				else:
					pbar.set_description("Removing {}".format(img))
			self.image_list = self.temp
			print('read {} images'.format(len(self.image_list)))
		
	def __len__(self):
		return len(self.image_list)

	def __getitem__(self, idx):
		img_name = os.path.join(self.root_dir, self.image_list[idx])
		data = h5py.File(img_name, 'r')
		image = np.array(data['image'], dtype=np.float32) 
		if self.read_label:
			label = np.array(data['label'])
			sample = {'image': image, 'label': label}
		else:
			sample = {'image': image}
		data.close()

		if self.transform:
			sample = self.transform(sample)
		
		return sample
