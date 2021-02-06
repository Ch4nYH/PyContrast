from .paths import get_paths
from .transforms import build_transforms
from .dataset import DatasetInstance, DatasetInstanceWithSSIM
from .puzzle_transforms import RandomCrop, ToTensor
import torchvision.transforms as transforms

def build_dataset(dataset, data_root, train_list, sampling = 'default', ssim = False, jigsaw=False):
    train_roots, train_lists, test_roots, test_lists = get_paths(dataset, data_root, train_list)
    train_transform, test_transform = build_transforms(sampling = sampling)

    if jigsaw:
        cell_size = 36 # size of volume we crop patch from
        patch_size = 32
        puzzle_config = 2 # 2 or 3 for 2X2X2 or 3X3X3 puzzle
        puzzle_num = puzzle_config ** 3
        jigsaw_transform = transforms.Compose([
            RandomCrop(cell_size, patch_size,
            puzzle_config, True),
            ToTensor(True)])

        train_dataset = DatasetInstance(train_lists, train_roots, dataset, transform=jigsaw_transform)
        test_dataset = DatasetInstance(test_lists, test_roots, dataset, transform=test_transform)
    elif not ssim:
        train_dataset = DatasetInstance(train_lists, train_roots, dataset, transform=train_transform)
        test_dataset = DatasetInstance(test_lists, test_roots, dataset, transform=test_transform)
    else:
        train_dataset = DatasetInstanceWithSSIM(train_lists, train_roots, dataset, transform=train_transform)
        test_dataset = DatasetInstanceWithSSIM(test_lists, test_roots, dataset, transform=test_transform)
    
    return train_dataset, test_dataset