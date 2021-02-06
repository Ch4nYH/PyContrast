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
        img_name = os.path.join(self.root_dir, 
                                self.image_list[idx].strip())
        image = np.load(img_name)['image'].astype(np.float32)

        # data processing
        sample = {'image': image}

        if self.transform:
            sample = self.transform(sample)

        return sample

class RandomCrop(object):
    def __init__(self, cell_size, patch_size, puzzle_config, flag_pair):
        self.cell_size = cell_size
        self.patch_size = patch_size
        self.puzzle_config = puzzle_config
        self.flag_pair = flag_pair

        self.puzzle_num = self.puzzle_config ** 3

    def __call__(self, sample):
        image = sample['image']
        (w, h, d) = image.shape

        w_c = np.random.randint(0, w - self.cell_size * self.puzzle_config + 1)
        h_c = np.random.randint(0, h - self.cell_size * self.puzzle_config + 1)
        d_c = np.random.randint(0, d - self.cell_size * self.puzzle_config + 1)

        patch_list = []
        for i_w in range(self.puzzle_config):
            for i_h in range(self.puzzle_config):
                for i_d in range(self.puzzle_config):
                    [w_p, h_p, d_p] = \
                        np.random.randint(0,
                                          self.cell_size - self.patch_size + 1,
                                          size=(3))

                    w_start = w_c + i_w * self.cell_size + w_p
                    h_start = h_c + i_h * self.cell_size + h_p
                    d_start = d_c + i_d * self.cell_size + d_p
                    patch_list.append(image[w_start : w_start + self.patch_size, \
                                            h_start : h_start + self.patch_size, \
                                            d_start : d_start + self.patch_size])

        u_label = np.random.permutation(self.puzzle_num)
        patch_list_disordered = [patch_list[i] for i in list(u_label)]

        image = np.stack(patch_list_disordered, 0)

        if self.flag_pair:
            b_label = np.zeros((self.puzzle_num * (self.puzzle_num - 1) / 2), dtype="int64")

            index = 0
            for i in range(self.puzzle_num):
                for j in xrange(i + 1, self.puzzle_num):
                    gap = u_label[j] - u_label[i]
                    if   gap ==  1:
                        b_label[index] = 0
                    elif gap == -1:
                        b_label[index] = 1
                    elif gap ==  self.puzzle_config:
                        b_label[index] = 2
                    elif gap == -self.puzzle_config:
                        b_label[index] = 3
                    elif gap ==  self.puzzle_config ** 2:
                        b_label[index] = 4
                    elif gap == -self.puzzle_config ** 2:
                        b_label[index] = 5
                    else:
                        b_label[index] = 6
                    index += 1
            return {'image' : image, 'u_label' : u_label, 'b_label' : b_label}
        else:
            return {'image' : image, 'u_label' : u_label}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, flag_pair):
        self.flag_pair = flag_pair

    def __call__(self, sample):
        image, u_label = sample['image'], sample['u_label']
        image = image.reshape(image.shape[0], 1, 
                              image.shape[1], image.shape[2], image.shape[3])
        if self.flag_pair:
            b_label = sample['b_label']
            return {'image'   : torch.from_numpy(image), \
                    'u_label' : torch.from_numpy(u_label), \
                    'b_label' : torch.from_numpy(b_label)}
        else: 
            return {'image'   : torch.from_numpy(image), \
                    'u_label' : torch.from_numpy(u_label)}
