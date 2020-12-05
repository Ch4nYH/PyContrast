from .paths import get_paths
from .transforms import build_transforms
from .dataset import DatasetInstance, DatasetInstanceWithSSIM

def build_dataset(dataset, data_root, train_list, sampling = 'default', ssim = False):
    train_roots, train_lists, test_roots, test_lists = get_paths(dataset, data_root, train_list)
    train_transform, test_transform = build_transforms(sampling = sampling)
    if not ssim:
        train_dataset = DatasetInstance(train_lists, train_roots, dataset, transform=train_transform)
        test_dataset = DatasetInstance(test_lists, test_roots, dataset, transform=test_transform)
    else:
        train_dataset = DatasetInstanceWithSSIM(train_lists, train_roots, dataset, transform=train_transform)
        test_dataset = DatasetInstanceWithSSIM(test_lists, test_roots, dataset, transform=test_transform)
    
    return train_dataset, test_dataset