import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.autograd import Variable
import torch.nn.functional as F
import random

class NIHDataset(Dataset):
    """ NIH Dataset """

    def __init__(self, list_file, root_dir, transform=None):
        """
        Args:
        list_file (string) : List of the image files
        root_dir (string) : Directory of images
        transform (callable, optional) : Optional transform to be applied on a sample
        """
        self.image_list = open(list_file).readlines()
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_list[idx].strip())
        #img_name = os.path.join(self.root_dir, self.image_list[idx].split('.')[0] + '.npz')
        image = np.load(img_name)['image'].astype(np.float32)
        label = np.load(img_name)['label'].astype(np.float32)

        # data processing

        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
        output_size (int): Desired output size
    """
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        (w, h, d) = image.shape
        if w < self.output_size[0]:
            w1 = 0
            sw = w
        else:
            w1 = np.random.randint(0, w - self.output_size[0])
            sw = self.output_size[0]

        if h < self.output_size[1]:
            h1 = 0
            sh = h
        else:
            h1 = np.random.randint(0, h - self.output_size[1])
            sh = self.output_size[1]

        if d < self.output_size[2]:
            d1 = 0
            sd = d
        else:
            d1 = np.random.randint(0, d - self.output_size[2])
            sd = self.output_size[2]
        
        label = label[w1:w1+sw, h1:h1+sh, d1:d1+sd]
        image = image[w1:w1+sw, h1:h1+sh, d1:d1+sd]

        return {'image' : image, 'label' : label}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
        label = label.reshape(1, label.shape[0], label.shape[1], label.shape[2])
        return {'image':torch.from_numpy(image), 'label':torch.from_numpy(label)}