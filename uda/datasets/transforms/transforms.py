import copy
import torch
import random
import torchvision
import numpy as np

from scipy.ndimage.filters import gaussian_filter, gaussian_gradient_magnitude

np.random.seed(42)
random.seed(42)

class RandomContrast(object):
    def __init__(self, contrast_range=(0.75, 1.25), preserve_range=True, per_channel=True):
        self.contrast_range = contrast_range
        self.preserve_range = preserve_range
        self.per_channel = per_channel
        
    def __call__(self, sample):
        
        image = sample['image'] 
        if not self.per_channel:
            mn = image.mean()
            if self.preserve_range:
                minm = image.min()
                maxm = image.max()
            if np.random.random() < 0.5 and self.contrast_range[0] < 1:
                factor = np.random.uniform(self.contrast_range[0], 1)
            else:
                factor = np.random.uniform(max(self.contrast_range[0], 1), self.contrast_range[1])
            image = (image - mn) * factor + mn
            if self.preserve_range:
                image[image < minm] = minm
                image[image > maxm] = maxm
        else:
            for c in range(image.shape[0]):
                mn = image[c].mean()
                if self.preserve_range:
                    minm = image[c].min()
                    maxm = image[c].max()
                if np.random.random() < 0.5 and self.contrast_range[0] < 1:
                    factor = np.random.uniform(self.contrast_range[0], 1)
                else:
                    factor = np.random.uniform(max(self.contrast_range[0], 1), self.contrast_range[1])
                image[c] = (image[c] - mn) * factor + mn
                if self.preserve_range:
                    image[c][image[c] < minm] = minm
                    image[c][image[c] > maxm] = maxm
                    
        sample['image'] = image
        return sample
class RandomCropSlices(object):
    def __init__(self, output_size, sample_num=4, pad=32, is_binary=True):
        self.output_size = output_size
        self.sample_num = sample_num
        self.pad = pad
        self.is_binary = is_binary

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        assert image.shape == label.shape
        if self.is_binary and label.max() > 1:
            label[label > 1] = 0
        if image.shape[0] < 256 or image.shape[1] < self.output_size or image.shape[2] < self.output_size: 
            pad_x = max(0, 256 - image.shape[0] + 1)
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
        output_image = np.zeros((self.sample_num, 64, self.output_size, self.output_size))
        output_label = np.zeros((self.sample_num, 64, self.output_size, self.output_size))
        for i in range(self.sample_num):
            #w1 = np.random.randint(bbox[0][0], bbox[0][1] - self.output_size + 1)
            w = 256
            h1 = np.random.randint(bbox[1][0], bbox[1][1] - self.output_size + 1)
            d1 = np.random.randint(bbox[2][0], bbox[2][1] - self.output_size + 1)
            output_image[i] = image[i * w // 4:i * w // 4 + w // 4, h1:h1+self.output_size, d1:d1+self.output_size]
            output_label[i] = label[i * w // 4:i * w // 4 + w // 4, h1:h1+self.output_size, d1:d1+self.output_size]
        return {'image': output_image, 'label': output_label, 'cropped_image': output_image}


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
      output_size (int): Desired output size
    """

    def __init__(self, output_size, sample_num=1, pad=32, is_binary=True):
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
        
        return {'image': output_image, 'label': output_label, 'cropped_image': output_image}


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
        return sample

    def _rotate(self, x, y):
        degree = random.choice([0, 1, 2, 3])
        x, y = np.rot90(x, degree, (1, 2)), np.rot90(y, degree, (1, 2))
        return x, y

class GaussianNoise(object):
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image += np.random.randn(*image.shape) / 5
        sample['image'] = image
        sample['label'] = label
        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        
        for key in sample.keys():
            image = sample[key]
            print(image.shape)
            if image.ndim >= 5:                                                     # already has channel dim
                image = torch.from_numpy(image.astype(np.float32))
            else:
                image = torch.from_numpy(image.astype(np.float32)).unsqueeze(1)     # Crop_num, 1, h, w, l

            sample[key] = image

        return sample

def get_range_val(value, rnd_type="uniform"):
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 2:
            if value[0] == value[1]:
                n_val = value[0]
            else:
                orig_type = type(value[0])
                if rnd_type == "uniform":
                    n_val = random.uniform(value[0], value[1])
                elif rnd_type == "normal":
                    n_val = random.normalvariate(value[0], value[1])
                n_val = orig_type(n_val)
        elif len(value) == 1:
            n_val = value[0]
        else:
            raise RuntimeError("value must be either a single vlaue or a list/tuple of len 2")
        return n_val
    else:
        return value
    
class GaussianBlur(object):
    def __init__(self, sigma_range=(1, 5), per_channel = True, p_per_channel = 1):
        self.sigma_range = sigma_range
        self.per_channel = per_channel
        self.p_per_channel = p_per_channel
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if not self.per_channel:
            sigma = get_range_val(self.sigma_range)
        for c in range(image.shape[0]):
            if np.random.uniform() <= self.p_per_channel:
                if self.per_channel:
                    sigma = get_range_val(self.sigma_range)
                image[c] = gaussian_filter(image[c], sigma, order=0)
                
        sample['image'] = image
        sample['label'] = label
        return sample


# Begin Jigsaw Transformations

class RandomCropJigsaw(object):
    def __init__(self, cell_size=70, patch_size=64, puzzle_config=2, flag_pair=True, is_binary=True):
        self.cell_size = cell_size
        self.patch_size = patch_size
        self.puzzle_config = puzzle_config
        self.flag_pair = flag_pair

        self.puzzle_num = self.puzzle_config ** 3
        self.is_binary = is_binary
    def __call__(self, sample):
        
        image = sample['image']
        label = sample['label']
        if self.is_binary:
            label[label > 1] = 0
        (w, h, d) = image.shape
        #print(w, h, d)
        w_c = np.random.randint(0, w - self.cell_size * self.puzzle_config + 1)
        h_c = np.random.randint(0, h - self.cell_size * self.puzzle_config + 1)
        d_c = np.random.randint(0, d - self.cell_size * self.puzzle_config + 1)

        patch_list = []
        label_list = []
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
                    label_list.append(label[w_start : w_start + self.patch_size, \
                                            h_start : h_start + self.patch_size, \
                                            d_start : d_start + self.patch_size])

        u_label = np.random.permutation(self.puzzle_num)
        patch_list_disordered = [patch_list[i] for i in list(u_label)]
        label_list_disordered = [label_list[i] for i in list(u_label)]


        image = np.stack(patch_list_disordered, 0)
        label = np.stack(label_list_disordered, 0)
        if self.flag_pair:
            b_label = np.zeros(int(self.puzzle_num * (self.puzzle_num - 1) / 2), dtype="int64")

            index = 0
            for i in range(self.puzzle_num):
                for j in range(i + 1, self.puzzle_num):
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
            return {'image' : image, 'label': label, 'u_label' : u_label, 'b_label' : b_label}
        else:
            return {'image' : image, 'label': label, 'u_label' : u_label}
