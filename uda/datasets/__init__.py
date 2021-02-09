from .paths import get_paths
from .transforms import build_transforms, build_jigsaw_transform
from .dataset import DatasetInstance, DatasetInstanceWithSSIM, DatasetInstanceJigsaw
from .puzzle_transforms import RandomCrop, ToTensor
import torchvision.transforms as transforms

def build_dataset(dataset, data_root, train_list, sampling = 'default', ssim = False, jigsaw=False):
    train_roots, train_lists, test_roots, test_lists = get_paths(dataset, data_root, train_list)

    if jigsaw:
        train_transform, test_transform = build_jigsaw_transform()
        train_dataset = DatasetInstanceJigsaw(train_lists, train_roots, dataset, transform=train_transform)
        test_dataset = DatasetInstanceJigsaw(test_lists, test_roots, dataset, transform=test_transform)
    elif not ssim:
        train_transform, test_transform = build_transforms(sampling = sampling)
        train_dataset = DatasetInstance(train_lists, train_roots, dataset, transform=train_transform)
        test_dataset = DatasetInstance(test_lists, test_roots, dataset, transform=test_transform)
    else:
        train_transform, test_transform = build_transforms(sampling = sampling)
        train_dataset = DatasetInstanceWithSSIM(train_lists, train_roots, dataset, transform=train_transform)
        test_dataset = DatasetInstanceWithSSIM(test_lists, test_roots, dataset, transform=test_transform)
    
    return train_dataset, test_dataset