import torchvision
from .transforms import RandomRotate, GaussianNoise, RandomContrast, ToTensor, RandomTranspose, \
    RandomCropSlices, RandomCrop


def build_transforms(pretrain = False):
    if pretrain:
        train_transforms = torchvision.transforms.Compose([
                        RandomCropSlices(64, 4, pad=-1, is_binary=True),
                        RandomRotate(),
                        GaussianNoise(),
                        RandomContrast(),
                        ToTensor()])

        test_transforms = torchvision.transforms.Compose([
                        RandomCrop(64, 8, pad=48, is_binary=True),
                        ToTensor()])
    else:
        train_transforms = torchvision.transforms.Compose([
                        RandomCrop(64, 8, pad=-1, is_binary=True),
                        RandomTranspose(),
                        RandomRotate(),
                        GaussianNoise(),
                        RandomContrast(),
                        ToTensor()])
        test_transforms = torchvision.transforms.Compose([
                        RandomCrop(64, 8, pad=48, is_binary=True),
                        ToTensor()])
    return train_transforms, test_transforms