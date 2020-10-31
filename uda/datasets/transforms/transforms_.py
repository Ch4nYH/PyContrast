import torch
import copy
import random
import numpy as np


try:  # SciPy >= 0.19
    from scipy.special import comb, logsumexp
except ImportError:
    from scipy.misc import comb, logsumexp


class RandomZoomAndScale(object):
    def __init__(self, output_size, sample_num=1):
        self.output_size = output_size
        self.sample_num = sample_num
        self.pad = pad
        self.is_binary = is_binary
    
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        assert image.shape == label.shape
        if self.is_binary and label.max() > 1:
            label[label > 1] = 0

        size_0, size_1, size_2 = image.shape
        size_out_0 = np.random.randint(size_0 * 2.0 / 3, size_0)
        size_out_1 = np.random.randint(size_1 * 2.0 / 3, size_1)
        size_out_2 = np.random.randint(size_2 * 2.0 / 3, size_2)

        start_0 = np.random.randint(0, size_0 - size_out_0)
        start_1 = np.random.randint(0, size_1 - size_out_1)
        start_2 = np.random.randint(0, size_2 - size_out_2)

        cropped_image = image[start_0:start_0+size_out_0, start_1:start_1:size_1, start_2:start_2+size_2]
        cropped_label = label[start_0:start_0+size_out_0, start_1:start_1:size_1, start_2:start_2+size_2]

        resize_img = resize(cropped_image, (self.output_size, self.output_size, self.output_size))
        resize_label = resize(cropped_label, (self.output_size, self.output_size, self.output_size))

        return {'image': resize_img, 'label': resize_label}

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


class RandomFlip(object):
    '''
    Randomly flip the image
    Args:
        prob: the probability to apply this transform
    '''

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if image.ndim == 3:
            if random.random() < self.prob:
                image, label = self._flip(image, label)
        elif image.ndim == 4:
            for i in range(len(image)):
                image[i], label[i] = self._flip(image[i], label[i])
        else:
            print('dim error')
            exit(-1)

        sample['image'] = image
        sample['label'] = label
        for key in sample:
            if key not in ['image', 'label']:
                sample[key] = copy.deepcopy(sample[key])
        return sample

    def _flip(self, image, label):
        degree = random.choice([0, 1, 2])
        image = np.flip(image, axis=degree)
        label = np.flip(label, axis=degree)
        return image, label


class PositionEncoding(object):
    '''
    Add position encoding to the image
    I0(i, j, k) is the intensity of the original image
    I1(i, j, k) = lambda * i / (W - 1)
    I2(i, j, k) = lambda * j / (H - 1)
    I3(i, j, k) = lambda * k / (L - 1)
    Args:
        prob: the probability to apply this transform
    '''

    def __init__(self, weight=1.0):
        self.weight = weight

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if image.ndim == 3:
            image = image[np.newaxis]
            image = np.concatenate((image, self._construct_PE(image)), axis=0)
        elif image.ndim == 4:
            image = image[:, np.newaxis]
            pos_enc = self._construct_PE(image)
            pos_enc = pos_enc[np.newaxis].repeat(image.shape[0], axis=0)
            image = np.concatenate((image, pos_enc), axis=1)
        else:
            print('dim error')
            exit(-1)

        sample['image'] = image
        sample['label'] = label
        for key in sample:
            if key not in ['image', 'label']:
                sample[key] = copy.deepcopy(sample[key])
        return sample

    def _construct_PE(self, image):
        I1 = np.arange(image.shape[-3]).astype(np.float) / (image.shape[-3] - 1)
        I1 = I1[:, np.newaxis, np.newaxis]
        I1 = np.tile(I1, (1, image.shape[-2], image.shape[-1]))
        I2 = np.arange(image.shape[-2]).astype(np.float) / (image.shape[-2] - 1)
        I2 = I2[np.newaxis, :, np.newaxis]
        I2 = np.tile(I2, (image.shape[-3], 1, image.shape[-1]))
        I3 = np.arange(image.shape[-1]).astype(np.float) / (image.shape[-1] - 1)
        I3 = I3[np.newaxis, np.newaxis, :]
        I3 = np.tile(I3, (image.shape[-3], image.shape[-2], 1))

        position_encoding = np.stack([I1, I2, I3]) * self.weight

        return position_encoding


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


class NonLinear(object):
    '''
    Apply a non-linear function to the image
    Args:
        prob: the probability to apply this transform
    '''

    def __init__(self, prob=0.6):
        self.prob = prob

    def __call__(self, sample):
        if random.random() >= self.prob:
            return sample
        x = sample['image']
        points = [[0, 0], [random.random(), random.random()], [random.random(), random.random()], [1, 1]]
        xpoints = [p[0] for p in points]
        ypoints = [p[1] for p in points]
        xvals, yvals = self.bezier_curve(points, nTimes=100000)
        if random.random() < 0.5:
            # Half change to get flip
            xvals = np.sort(xvals)
        else:
            xvals, yvals = np.sort(xvals), np.sort(yvals)
        nonlinear_x = np.interp(x, xvals, yvals)
        sample['image'] = nonlinear_x
        return sample

    def bernstein_poly(self, i, n, t):
        """
         The Bernstein polynomial of n, i as a function of t
        """
        return comb(n, i) * (t ** (n - i)) * (1 - t) ** i

    def bezier_curve(self, points, nTimes=1000):
        """
           Given a set of control points, return the
           bezier curve defined by the control points.

           Control points should be a list of lists, or list of tuples
           such as [ [1,1],
                     [2,3],
                     [4,5], ..[Xn, Yn] ]
            nTimes is the number of time steps, defaults to 1000

            See http://processingjs.nihongoresources.com/bezierinfo/
        """

        nPoints = len(points)
        xPoints = np.array([p[0] for p in points])
        yPoints = np.array([p[1] for p in points])

        t = np.linspace(0.0, 1.0, nTimes)

        polynomial_array = np.array([self.bernstein_poly(i, nPoints - 1, t) for i in range(0, nPoints)])

        xvals = np.dot(xPoints, polynomial_array)
        yvals = np.dot(yPoints, polynomial_array)

        return xvals, yvals


class LocalPixelShuffle(object):
    '''
    Apply a local pixel shuffle transform to the image
    Args:
        prob: the probability to apply this transform
    '''

    def __init__(self, prob=0.1):
        self.prob = prob

    def __call__(self, sample):
        if random.random() >= self.prob:
            return sample
        x = sample['image']
        image_temp = copy.deepcopy(x)
        orig_image = copy.deepcopy(x)
        img_rows, img_cols, img_deps = x.shape
        num_block = 500
        for _ in range(num_block):
            block_noise_size_x = random.randint(1, img_rows // 10)
            block_noise_size_y = random.randint(1, img_cols // 10)
            block_noise_size_z = random.randint(1, img_deps // 10)
            noise_x = random.randint(0, img_rows - block_noise_size_x)
            noise_y = random.randint(0, img_cols - block_noise_size_y)
            noise_z = random.randint(0, img_deps - block_noise_size_z)
            window = orig_image[noise_x:noise_x + block_noise_size_x,
                     noise_y:noise_y + block_noise_size_y,
                     noise_z:noise_z + block_noise_size_z,
                     ]
            window = window.flatten()
            np.random.shuffle(window)
            window = window.reshape((block_noise_size_x,
                                     block_noise_size_y,
                                     block_noise_size_z))
            image_temp[noise_x:noise_x + block_noise_size_x,
            noise_y:noise_y + block_noise_size_y,
            noise_z:noise_z + block_noise_size_z] = window
        local_shuffling_x = image_temp
        sample['image'] = local_shuffling_x
        return sample


class Painting(object):
    '''
    Apply a non-linear function to the image
    Args:
        prob: the probability to apply this transform
    '''

    def __init__(self, prob=0.6, in_painting_prob=0.2):
        self.prob = prob
        self.in_painting_prob = in_painting_prob

    def __call__(self, sample):
        image = sample['image']
        if random.random() < self.prob:
            if random.random() < self.in_painting_prob:
                # Inpainting
                image = self.in_painting(image)
            else:
                # Outpainting
                image = self.out_painting(image)
        sample['image'] = image
        return sample

    def in_painting(self, x):
        img_rows, img_cols, img_deps = x.shape
        block_noise_size_x = random.randint(10, 20)
        block_noise_size_y = random.randint(10, 20)
        block_noise_size_z = random.randint(10, 20)
        noise_x = random.randint(3, img_rows - block_noise_size_x - 3)
        noise_y = random.randint(3, img_cols - block_noise_size_y - 3)
        noise_z = random.randint(3, img_deps - block_noise_size_z - 3)
        x[
        noise_x:noise_x + block_noise_size_x,
        noise_y:noise_y + block_noise_size_y,
        noise_z:noise_z + block_noise_size_z] = random.random()
        return x

    def out_painting(self, x):
        img_rows, img_cols, img_deps = x.shape
        block_noise_size_x = img_rows - random.randint(10, 20)
        block_noise_size_y = img_cols - random.randint(10, 20)
        block_noise_size_z = img_deps - random.randint(10, 20)
        noise_x = random.randint(3, img_rows - block_noise_size_x - 3)
        noise_y = random.randint(3, img_cols - block_noise_size_y - 3)
        noise_z = random.randint(3, img_deps - block_noise_size_z - 3)
        image_temp = copy.deepcopy(x)
        x = np.random.rand(x.shape[0], x.shape[1], x.shape[2],) * 1.0  # 0 to 1
        x[
        noise_x:noise_x + block_noise_size_x,
        noise_y:noise_y + block_noise_size_y,
        noise_z:noise_z + block_noise_size_z] = image_temp[noise_x:noise_x + block_noise_size_x,
                                                noise_y:noise_y + block_noise_size_y,
                                                noise_z:noise_z + block_noise_size_z]
        return x


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
