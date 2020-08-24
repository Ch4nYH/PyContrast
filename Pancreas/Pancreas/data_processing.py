import os
import h5py
import numpy as np
import json

# data path
data_path = '/export/ccvl12b/datasets/nih_pancreas' # npy path
save_path = os.path.join(data_path, 'nih_pad32/')

save_list_path = 'train.list' # where to save your training lists

pad = [32, 32, 32] # ROI padding
rand_num = 16 # controls how many we sample from whole CT images , currently we adopt foreground: background = 1:1

min_val = 0
max_val = 255

if not os.path.exists(save_path):
    os.makedirs(save_path)

def load_data(path):
    data = h5py.File(path, 'r')
    image = np.array(data['raw']).astype(np.float32)
    label = np.array(data['label'])

    image[image < min_val] = min_val
    image[image > max_val] = max_val
    image = (image - min_val) / (max_val - min_val)

    # get the bounding box from label plus padding
    tempL = np.nonzero(label)
    bbox = [[max(0, np.min(tempL[0])-pad[0]), min(label.shape[0], np.max(tempL[0])+pad[0])], \
            [max(0, np.min(tempL[1])-pad[1]), min(label.shape[1], np.max(tempL[1])+pad[1])], \
            [max(0, np.min(tempL[2])-pad[2]), min(label.shape[2], np.max(tempL[2])+pad[2])]]

    # crop random sample on whole image
    w, h, d = image.shape
    w_l, h_l, d_l = bbox[0][1] - bbox[0][0], bbox[1][1] - bbox[1][0], bbox[2][1] - bbox[2][0]
    image_list = []
    label_list = []
    print(image.shape)
    print(bbox)
    for i in range(rand_num):
        w1 = np.random.randint(0, w - w_l+1)
        h1 = np.random.randint(0, h - h_l+1)
        d1 = np.random.randint(0, d - d_l+1)
        image_list.append(image[w1:w1+w_l, h1:h1+h_l, d1:d1+d_l])
        label_list.append(label[w1:w1+w_l, h1:h1+h_l, d1:d1+d_l])
    
    image = image[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]]
    label = label[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]]
    return image, label, image_list, label_list

f = open(save_list_path, 'w')
img_dir = os.path.join(data_path, 'raw/')
for idx in range(1,83):
    path = os.path.join(img_dir, "%03d.npy.h5" % idx)
    image, label, image_list, label_list = load_data(image_path, label_path)

    for i in range(rand_num):
        image_save_path = os.path.join(save_path, str(idx) + '_' + str(i) + '.npz')
        np.savez(image_save_path, image=image, label=label)
        f.write(str(idx) + '_' + str(i) + '.npz' + '\n')
        print(image_save_path)
    
    for i in range(rand_num):
        image_save_path = os.path.join(save_path, str(idx) + '_r' + str(i) + '.npz')
        np.savez(image_save_path, image=image_list[i], label=label_list[i])
        f.write(str(idx) + '_r' + str(i) + '.npz' + '\n')
        print(image_save_path)

f.close()