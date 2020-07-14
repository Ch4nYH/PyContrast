import torch
import copy
import numpy as np
import torch.nn.functional as F
from pdb import set_trace as bp

class ExtendLength(object):
    def __init__(self, desire_len=512):
        self.desire_len = desire_len

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        expected_size = (self.desire_len,) + image.shape[1:]
        image = image[np.newaxis, np.newaxis, :].astype(np.float32)
        label = label[np.newaxis, np.newaxis, :].astype(np.float32)
        image = F.interpolate(torch.from_numpy(image), size=expected_size, mode='trilinear', align_corners=False).numpy()[0, 0, :, :, :]
        label = F.interpolate(torch.from_numpy(label), size=expected_size, mode='trilinear', align_corners=False).numpy()[0, 0, :, :, :]

        sample['image'] = image
        sample['label'] = label
        return sample


class CropLabel(object):
  """
  Crop randomly the image in a sample
  Args:
    output_size (int): Desired output size
  """

  def __init__(self, pad=16, is_binary = True):
    self.pad = pad
    self.is_binary = is_binary
  def __call__(self, sample):
    image, label = sample['image'], sample['label']
    if self.is_binary and label.max() > 1:
        label[label > 1] = 0
    tempL = np.nonzero(label)
    try:
        bbox = [[max(0, np.min(tempL[0]) - self.pad), min(label.shape[0], np.max(tempL[0]) + 1 + self.pad)],
            [max(0, np.min(tempL[1]) - self.pad), min(label.shape[1], np.max(tempL[1]) + 1 + self.pad)],
            [max(0, np.min(tempL[2]) - self.pad), min(label.shape[2], np.max(tempL[2]) + 1 + self.pad)]]
    except ValueError:
        print(label)
        print(tempL)
        print(label.max())
        print(label.min())

    # crop random sample on whole image
    output_image = image[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]]
    output_label = label[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]]
    return {'image': output_image, 'label': output_label}


class SamplePuzzle(object):
    '''
    Sample puzzle_config * puzzle_config cubes of size (patch_size, patch_size, patch_size)
    '''
    def __init__(self, cell_size, patch_size, puzzle_config=3, flag_pair=False, upscale_factor=-1):
        self.cell_size = cell_size
        self.patch_size = patch_size
        self.puzzle_config = puzzle_config
        self.flag_pair = flag_pair

        if upscale_factor > 1:
            self.patch_length = int(patch_size / upscale_factor)
            self.cell_length = int(cell_size / upscale_factor)
        else:
            self.patch_length = patch_size
            self.cell_length = cell_size

        self.puzzle_num = self.puzzle_config ** 3
        self.patch_vol = self.patch_size ** 2 * self.patch_length

    def check_shape(self, image):
        (w, h, d) = image.shape
        if w - self.cell_length * self.puzzle_config + 1 > 0 \
                and h - self.cell_size * self.puzzle_config + 1 > 0 \
                and d - self.cell_size * self.puzzle_config + 1 > 0:
            return True, image.shape

        desired_w = max(self.cell_length * self.puzzle_config, w)
        desired_h = max(self.cell_size * self.puzzle_config, h)
        desired_d = max(self.cell_size * self.puzzle_config, d)
        desired_shape = (desired_w, desired_h, desired_d)
        return False, desired_shape

    def __call__(self, sample):
        image = sample['image']
        label = sample['label'] if 'label' in sample else None
        check_ok, desired_shape = self.check_shape(image)

        if not check_ok:
            image = np.reshape(image, (1, 1,) + image.shape).astype(np.float32)
            image = F.interpolate(torch.from_numpy(image), size=desired_shape, mode='trilinear',
                                  align_corners=False).numpy()[0, 0, :, :, :]
            if label is not None:
                label = np.reshape(label, (1, 1,) + label.shape).astype(np.float32)
                label = F.interpolate(torch.from_numpy(label), size=desired_shape, mode='nearest').numpy()[0, 0, :, :, :]
        (w, h, d) = image.shape
        w_c = np.random.randint(0, w - self.cell_length * self.puzzle_config + 1)
        h_c = np.random.randint(0, h - self.cell_size * self.puzzle_config + 1)
        d_c = np.random.randint(0, d - self.cell_size * self.puzzle_config + 1)

        patch_list = []
        label_list = []
        pos_list = []
        neg_list = []
        idx = 0
        for i_w in range(self.puzzle_config):
            for i_h in range(self.puzzle_config):
                for i_d in range(self.puzzle_config):
                    w_p = np.random.randint(0, self.cell_length - self.patch_length + 1)
                    [h_p, d_p] = np.random.randint(0, self.cell_size - self.patch_size + 1, size=(2))

                    w_start = w_c + i_w * self.cell_length + w_p
                    h_start = h_c + i_h * self.cell_size + h_p
                    d_start = d_c + i_d * self.cell_size + d_p
                    patch_list.append(image[w_start : w_start + self.patch_length,
                                            h_start : h_start + self.patch_size,
                                            d_start : d_start + self.patch_size])
                    if label is not None:
                        label_patch = label[w_start : w_start + self.patch_length,
                                            h_start : h_start + self.patch_size,
                                            d_start : d_start + self.patch_size]
                        label_list.append(label_patch)
                        if (label_patch > 0).sum() > 0.1 * self.patch_vol:
                            pos_list.append(idx)
                        else:
                            neg_list.append(idx)
                    idx += 1
        u_label = np.random.permutation(self.puzzle_num)
        patch_list_disordered = [patch_list[i] for i in list(u_label)]
        puzzles = np.stack(patch_list_disordered, 0)

        if label is not None:
            label_list_disordered = [label_list[i] for i in list(u_label)]
            label = np.stack(label_list_disordered, 0)

        if self.flag_pair:
            b_label = np.zeros((self.puzzle_num * (self.puzzle_num - 1) / 2), dtype="int64")

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
            sample =  {'image' : puzzles, 'u_label' : u_label, 'b_label' : b_label}
        else:
            sample =  {'image' : puzzles, 'u_label' : u_label}

        if label is not None:
            sample['label'] = label
        return sample


class DownsampleLength(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, upscale_factor=4, dim=0):
        self.upscale_factor = upscale_factor

    def __call__(self, sample):
        puzzles = sample['image']
        hr_images = copy.deepcopy(puzzles)
        if puzzles.ndim == 3:
            puzzles = puzzles[::self.upscale_factor, :, :]
        elif puzzles.ndim == 4:
            puzzles = puzzles[:, ::self.upscale_factor, :, :]

        sample['hr_image'] = hr_images
        sample['image'] = puzzles

        return sample


class PuzzleToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, flag_pair=False):
        self.flag_pair = flag_pair

    def __call__(self, sample):
        puzzles, u_label = sample['image'], sample['u_label']
        label = sample['label'] if 'label' in sample else None
        puzzles = np.expand_dims(puzzles, axis=-4)

        if label is not None:
            label = np.expand_dims(label, axis=-4)
            sample['label'] = torch.from_numpy(label.astype(np.float32))

        sample['image'] = torch.from_numpy(puzzles)
        sample['u_label'] = torch.from_numpy(u_label)

        if self.flag_pair:
            b_label = sample['b_label']
            sample['b_label'] = torch.from_numpy(b_label)
        if 'hr_image' in sample:
            hr_image = sample['hr_image']
            hr_image = np.expand_dims(hr_image, axis=-4)
            sample['hr_image'] = torch.from_numpy(hr_image)
        return sample
