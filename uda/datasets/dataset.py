import os
import h5py
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

from .paths import get_paths
from .transforms.transforms import build_transforms
from .utils import ssim

import torch.nn.functional as F



class DatasetInstance(Dataset):

    def __init__(self, list_file, root_dir, transform=None, 
        need_non_zero_label = True, is_binary = False, jigsaw_transform = None, dataset='nih_pancreas'):
        self.image_list = open(list_file).readlines()
        self.image_list = [os.path.basename(line.strip()) for line in self.image_list]
        self.image_list = [line for line in self.image_list if line.endswith('.h5')]

        self.root_dir = root_dir
        self.transform = transform

        self.two_crop = True
        self.use_jigsaw = False #TODO
        self.dataset = dataset
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
        

        if self.dataset == 'synapse':
            image = torch.from_numpy(image)
            shape = list(image.shape)
            shape[0] *= 3
            image = image.reshape((1,1,)+image.shape)
            image = F.interpolate(image, size = tuple(shape), mode='trilinear')
            image = image[0,0,:,:,:]
            image = image.numpy()

        if self.transform is not None:
            sample = self.transform(sample_pre_transform)
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
    
    
def build_dataset(args):
    train_root, train_list, test_root, test_list = get_paths(args.dataset, args.data_root, args.train_list)
    train_transform, test_transform = build_transforms(args)
    if not args.ssim:
        train_dataset = DatasetInstance(train_list, train_root, transform=train_transform, dataset = args.dataset)

        test_dataset = DatasetInstance(test_list, test_root, transform=test_transform, dataset = args.dataset)
    else:
        train_dataset = DatasetInstanceWithSSIM(train_list, train_root, transform=train_transform, dataset = args.dataset)

        test_dataset = DatasetInstanceWithSSIM(test_list, test_root, transform=test_transform, dataset = args.dataset)
    
    return train_dataset, test_dataset