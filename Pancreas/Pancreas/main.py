import sys
import math
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import torch.nn.functional as F
from model import VNet # import your model here
from utils import dice, dice_loss
from data import DataLoader, NIHDataset, RandomCrop, ToTensor
import numpy as np

current_fold = 0

list_path = './lists/train_fd0.list'
train_data_path = '/export/ccvl12b/datasets/nih_pancreas/nih_pad32/'
test_data_path = '/export/ccvl12b/datasets/nih_pancreas/test'

test_list_path = './lists/test_fd0'

snapshot_path = 'models/'
snapshot_prefix = 'vnet_fd' + str(current_fold)
if not os.path.exists(snapshot_path):
  os.mkdir(snapshot_path)

os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]

max_iterations = 20000
patch_size = (96,96,96)
stride = 32 # sliding window testing stride
base_lr = 1e-2 # base learning rate

if __name__ == "__main__":
    net = VNet().cuda()
    
    if sys.argv[2] == 'train':
        net_parallel = nn.DataParallel(net)
        dataset = NIHDataset(list_file = list_path, 
                                root_dir = train_data_path,
                                transform = transforms.Compose([
                                RandomCrop(patch_size),
                                ToTensor(),
                                ]))
        dataloader = DataLoader(dataset, batch_size = 4,
                                shuffle=True, num_workers = 4)
        
        net_parallel.train()
        
        optimizer = optim.SGD(net_parallel.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.00004)
        iter_num = 0
        while True:
            for i_batch, sampled_batch in enumerate(dataloader):
                volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
                output = net_parallel(volume_batch)
                
                output = F.sigmoid(output)
                loss = dice_loss(output, label_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                iter_num = iter_num + 1
                if iter_num % 10 == 0:
                    print('iteration %d : loss : %f' % (iter_num, loss.item()))
                if iter_num % 5000 == 0:
                    torch.save(net.state_dict(), os.path.join(snapshot_path, snapshot_prefix + '_iteration_' + str(iter_num) + '.pth'))
                if iter_num >= max_iterations:
                    break
            if iter_num >= max_iterations:
                break

    if sys.argv[2] == 'test':
        load_snapshot_path = sys.argv[3]
        net.load_state_dict(torch.load(load_snapshot_path))
        net.eval()
        print("")
        print("#############################################")
        dices = []
        for filename in open(test_list_path).readlines():
            idx = int(filename.split('.')[0])
            data = h5py.File(os.path.join(test_data_path, '%03d.npy' % idx), 'r')
            image = np.array(data['raw']).astype('float') / 255.0
            label = np.array(data['label'])
            
            w, h, d = image.shape
            sx = math.floor((w - patch_size[0]) / stride) + 1
            sy = math.floor((h - patch_size[1]) / stride) + 1
            sz = math.floor((d - patch_size[2]) / stride) + 1
            score_map = np.zeros(image.shape).astype(np.float32)
            time_map = np.zeros(image.shape)
            for x in range(0, sx):
                xs = stride * x
                for y in range(0, sy):
                    ys = stride * y
                    for z in range(0, sz):
                        zs = stride * z
                        test_patch = image[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                        test_patch = test_patch.reshape(1, 1, test_patch.shape[0], test_patch.shape[1], test_patch.shape[2]).astype(np.float32)
                        test_patch = torch.from_numpy(test_patch).cuda()
                        output = net(test_patch)
                        output_up = F.sigmoid(output)
                        output = output_up.cpu().data.numpy()
                        output = output[0,0]
                        score_map[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] = score_map[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + output
                        time_map[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] = time_map[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + np.ones(output.shape)
            time_map[time_map==0] = 1
            score_map = score_map / time_map

            score_map = (score_map > 0.5).astype(np.int)
            dsc = dice(score_map, label)
            dices.append(dsc)
            print('case ' + filename.split('.')[0]  + ' dsc : %f' % dsc)
        print('mean dsc for fold %d : %f' % (current_fold, sum(dices) / len(dices)))
        print("##########################################")
        print("")

    else:
        print("No Such Arguments!")