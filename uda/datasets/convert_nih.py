import h5py
import sys
sys.path.append('.')
import os
import numpy as np
import argparse
import random
from datetime import datetime
from tqdm import tqdm
from datasets.utils import sort_nicely, process_slice

random.seed(1234)


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="process nih dataset")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Input data directory")
    parser.add_argument("--output_dir", type=str, default=None,
                        help='Output data directory')
    parser.add_argument("--ratio", type=float, default=0.8, help="training data ratio.")

    return parser.parse_args()


args = get_arguments()


def _read_img_and_label(directory, case_name):
    label_file = os.path.join(directory, 'labels', case_name)
    img_file = os.path.join(directory, 'images', case_name)

    assert os.path.exists(img_file) and os.path.exists(label_file)
    label = np.load(label_file)
    img = np.load(img_file)

    return img, label


def _process_dataset(directory, case_list, output_dir):
    """Process a complete data set and save it as a TFRecord.
    Args:
    name: string, unique identifier specifying the data set.
    directory: string, root path to the data set.
    num_shards: integer number of shards for this data set.
    labels_file: string, path to the labels file.
    """
    filenames = []

    for case_name in tqdm(case_list):
        img, label = _read_img_and_label(directory, case_name)
        img = process_slice(img)

        img = np.flip(img, axis=2)
        label = np.flip(label, axis=2)
        img = np.transpose(img, (2, 0, 1))
        label = np.transpose(label, (2, 0, 1))

        img = np.rot90(img, 3, (1, 2))
        label = np.rot90(label, 3, (1, 2))

        assert img.shape == label.shape, 'image shape is ({}, {}, {}) and label shape is ({}, {}, {})'.format(
            img.shape[0], img.shape[1], img.shape[2], label.shape[0], label.shape[1], label.shape[2])

        h5f = h5py.File(os.path.join(output_dir, os.path.basename(case_name) + '.h5'), 'w')
        h5f.create_dataset('raw', data=img)
        h5f.create_dataset('label', data=label)
        h5f.close()

    print('%s: Finished writing all %d cases in data set.\n' % (datetime.now(), len(case_list)))
    sys.stdout.flush()

    # write image names to a list
    with open(os.path.join(output_dir, '../{}.lst'.format(os.path.basename(output_dir))), 'w') as f:
        for case_names in filenames:
            for item in case_names:
                f.write('{}\n'.format(item))

    return filenames


def process_dataset(output_dir, all_cases, name):
    train_len = int(len(all_cases) * args.ratio)
    test_len = len(all_cases) - train_len

    print('Got {} cases for {}, split into {} train cases and {} test cases'.format(len(all_cases), name, train_len, test_len))

    random.shuffle(all_cases)

    train_cases = all_cases[:train_len]
    test_cases = all_cases[train_len:train_len + test_len]

    if not os.path.exists(output_dir):
        create_dirs = [os.path.join(output_dir, 'train'),
                       os.path.join(output_dir, 'test')]
        for c_dir in create_dirs:
            os.makedirs(c_dir)

    print('------------ processing train data for {} ------------'.format(name))
    _process_dataset(args.input_dir, train_cases, os.path.join(output_dir, 'train'))
    print('------------ processing test data for {} ------------'.format(name))
    _process_dataset(args.input_dir, test_cases, os.path.join(output_dir, 'test'))


def main():
    all_cases = os.listdir(os.path.join(args.input_dir, 'images'))
    all_cases = sort_nicely(all_cases)
    process_dataset(args.output_dir, all_cases, name='nih pancreas')


if __name__ == '__main__':
    main()

