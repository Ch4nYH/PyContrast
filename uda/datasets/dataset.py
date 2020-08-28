import os
import h5py
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

from .paths import get_paths
from .transforms import build_transforms

class DatasetInstance(Dataset):

    def __init__(self, list_file, root_dir, transform=None, 
        need_non_zero_label = True, is_binary = False, jigsaw_transform = None):
        self.image_list = open(list_file).readlines()
        self.image_list = [os.path.basename(line.strip()) for line in self.image_list]
        self.image_list = [line for line in self.image_list if line.endswith('.h5')]

        self.root_dir = root_dir
        self.transform = transform

        self.two_crop = True
        self.use_jigsaw = False #TODO

        print('read {} images'.format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        img_name = os.path.join(self.root_dir, self.image_list[index])
        data = h5py.File(img_name, 'r')
        image = np.array(data['image'], dtype=np.float32) 
        label = np.array(data['label'])
        
        data.close()
        sample_pre_transform = {'image': image, 'label': label}

        if self.transform is not None:
            sample = self.transform(sample_pre_transform)
            if self.two_crop:
                sample2 = self.transform(sample_pre_transform)
                sample['image_2'] = sample2['image']
                sample['label_2'] = sample2['label']
        else:
            img = image

        if self.use_jigsaw:
             jigsaw_img = self.jigsaw_transform(sample_pre_transform)


        sample['index'] = index
        return sample

def build_dataset(args):
    train_root, train_list, test_root, test_list = get_paths(args.dataset, args.data_root, args.train_list)
    train_transform, test_transform = build_transforms(args)

    train_dataset = DatasetInstance(train_list, train_root, transform=train_transform)

    test_dataset = DatasetInstance(test_list, test_root, transform=test_transform)
    
    return train_dataset, test_dataset