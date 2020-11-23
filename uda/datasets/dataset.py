import os
import h5py
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

from .paths import get_paths
from .transforms import build_transforms
from .utils import ssim

import torch.nn.functional as F



class DatasetInstance(Dataset):

    def __init__(self, list_files, root_dirs, dataset_names, transform=None, 
        need_non_zero_label = True, is_binary = False, jigsaw_transform = None, ):

        assert len(list_files) > 0, "Must provide lists!"
        self.image_list = []
        self.lengths = []
        for list_file in list_files:
            image_list = open(list_file).readlines()
            image_list = [os.path.basename(line.strip()) for line in image_list]
            image_list = [line for line in image_list if line.endswith('.h5')]
            self.lengths.append(len(image_list))
            self.image_list.extend(image_list)

        self.root_dir = {}
        for name, path in zip(dataset_names, root_dirs):
            self.root_dir[name] = path

        self.transform = transform

        self.two_crop = True
        self.use_jigsaw = False #TODO
        self.datasets = dataset_names
        
        self.data_name = []
        for (d, l) in zip(self.datasets, self.lengths):
            print('{}: {} images'.format(d, l))
            self.data_name.extend([d] * l)


    def __len__(self):

        return len(self.image_list)

    def __getitem__(self, index):
        img_name = os.path.join(self.root_dir[self.data_name[index]], self.image_list[index])
        data = h5py.File(img_name, 'r')
        image = np.array(data['image'], dtype=np.float32) 
        label = np.array(data['label'])
        
        data.close()
        sample_pre_transform = {'image': image, 'label': label}
        

        if self.data_name[index] == 'synapse':
            image = torch.from_numpy(image)
            shape = list(image.shape)
            shape[0] *= 3
            image = image.reshape((1,1,)+image.shape)
            image = F.interpolate(image, size = tuple(shape), mode='trilinear')
            image = image[0,0,:,:,:]
            image = image.numpy()

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

class DatasetInstanceWithSSIM(DatasetInstance):
    def __init__(self, list_file, root_dir, transform=None, 
        need_non_zero_label = True, is_binary = False, jigsaw_transform = None, num_of_samples=7, split='train', dataset='nih_pancreas'):
        super(DatasetInstanceWithSSIM, self).__init__(list_file, root_dir, transform, 
        need_non_zero_label, is_binary, jigsaw_transform, dataset=dataset)
        self.num_of_samples = num_of_samples
        self.split = split
    def __getitem__(self, index):
        img_name = os.path.join(self.root_dir, self.image_list[index])
        data = h5py.File(img_name, 'r')
        image = np.array(data['image'], dtype=np.float32) 
        label = np.array(data['label'])
        
        data.close()
        sample_pre_transform = {'image': image, 'label': label}

        if self.transform is not None:
            sample = self.transform(sample_pre_transform)
            best_ssim = 0
            best_i = 0
            best_j = 0
            if self.two_crop:
                new_samples = [self.transform(sample_pre_transform) for _ in range(self.num_of_samples)]
                new_samples.append(sample)
                for i in range(len(new_samples)):
                    for j in range(len(new_samples)):
                        if i != j:
                            sample_ssim = ssim(new_samples[i]['cropped_image'], new_samples[j]['cropped_image'])
                            if sample_ssim > best_ssim:
                                best_ssim = sample_ssim
                                best_i = i
                                best_j = j
                            
            sample['image'] = new_samples[best_i]['image']
            sample['label'] = new_samples[best_i]['label']
            sample['image_2'] = new_samples[best_j]['image']
            sample['label_2'] = new_samples[best_j]['label']
        else:
            img = image

        if self.use_jigsaw:
             jigsaw_img = self.jigsaw_transform(sample_pre_transform)


        sample['index'] = index
        return sample
    
    
