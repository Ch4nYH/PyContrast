import torchvision
from .transforms import RandomRotate, GaussianNoise, RandomContrast, ToTensor, RandomTranspose, \
    RandomCropSlices, RandomCrop, RandomCropJigsaw


def build_transforms(sampling = 'default'):
    if sampling == 'layerwise':
        print("using layerwise sampling")
        train_transforms = torchvision.transforms.Compose([
                        RandomCropSlices(64, 4, pad=-1, is_binary=True),
                        RandomRotate(),
                        GaussianNoise(),
                        RandomContrast(),
                        ToTensor()])

        test_transforms = torchvision.transforms.Compose([
                        RandomCrop(64, 8, pad=48, is_binary=True),
                        ToTensor()])
    elif sampling == 'default':
        print("using default sampling")
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
    else:
        raise ValueError("unsupported sampling method")
    return train_transforms, test_transforms

def build_jigsaw_transform():
    train_transforms = torchvision.transforms.Compose([
                        RandomCropJigsaw(),
                        RandomTranspose(),
                        RandomRotate(),
                        GaussianNoise(),
                        RandomContrast(),
                        ToTensor()])
    test_transforms = torchvision.transforms.Compose([
                        RandomCrop(64, 8, pad=48, is_binary=True),
                        ToTensor()])
    
    return train_transforms, test_transforms