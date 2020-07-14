import os
import numpy as np
import h5py
from tqdm import tqdm

root_path = None
ld_path = None
sv_path = None


def load_data(imglabelpath):
    print(imglabelpath)
    data = h5py.File(imglabelpath, 'r')
    image = data['raw']
    label = data['label']

    # change from dataset to array
    image = np.array(image).astype(np.float32)
    label = np.array(label).astype(dtype=np.uint8)

    ## do the zero mean and unit variance
    # only caculate the mean and variance values of the positive
    mean_val = np.mean(image[image > 0])
    std_val = np.std(image[image > 0])
    image = (image - mean_val) / std_val

    assert image.shape == label.shape
    return image, label


for split in ['train', 'test']:
    load_path = ld_path.format(split)
    save_path = sv_path.format(split)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    filenames = os.listdir(load_path)
    for i, file in enumerate(tqdm(filenames)):
        if file.endswith(".h5"):
            imglabelpath = os.path.join(load_path, file)
            img, label = load_data(imglabelpath)
            h5f = h5py.File(os.path.join(save_path, file), 'w')
            h5f.create_dataset('image', data=img)
            h5f.create_dataset('label', data=label)
            h5f.close()
