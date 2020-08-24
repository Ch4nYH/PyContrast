import sys
import math
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import torch.nn.functional as F

from vnet_puzzle import puzzlenet
from utils_puzzle import unary_loss, binary_loss
from data_puzzle import DataLoader, NIHDataset, RandomCrop, ToTensor
import numpy as np

train_list_path = './lists/train_fd0.list'
train_data_path = '/DATA/disk1/Pancreas/nih_pad20/'

test_data_path = '/DATA/disk1/Pancreas/'
test_list_path = './lists/train_fd0.list'

snapshot_path = 'models' # where to save your models
snapshot_prefix = 'puzzle'
if not os.path.exists(snapshot_path):
    os.mkdir(snapshot_path)

os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]

max_iterations = 20000
base_lr = 1e-2 # base learning rate
batch_size = 64

# puzzle arguments
cell_size = 36 # size of volume we crop patch from
patch_size = 32
puzzle_config = 2 # 2 or 3 for 2X2X2 or 3X3X3 puzzle
puzzle_num = puzzle_config ** 3
feature_len = 1024 # 
iter_num = 5
flag_pair = True


if __name__ == "__main__":
    net = puzzlenet(feature_len, puzzle_num, iter_num, flag_pair).cuda()
    #params = list(net.named_parameters())
    #for param in params:
    #    print(param[0])

    if sys.argv[2] == 'train':
        net_parallel = nn.DataParallel(net)
        dataset = NIHDataset(list_file = train_list_path, 
                                root_dir = train_data_path,
                                transform = transforms.Compose([
                                RandomCrop(cell_size, patch_size,
                                           puzzle_config, flag_pair),
                                ToTensor(flag_pair),
                                ]))

        dataloader = DataLoader(dataset, batch_size = batch_size,
                                shuffle=True, num_workers = 4)

        net_parallel.train()
        
        optimizer = optim.SGD(net_parallel.parameters(), 
                              lr=base_lr,
                              momentum=0.9,
                              weight_decay=0.00004)

        iter_num = 1
        while True:
            for i_batch, sampled_batch in enumerate(dataloader):
                volume_batch, u_label_batch = sampled_batch['image'], \
                                              sampled_batch['u_label']
                volume_batch, u_label_batch = volume_batch.cuda(), \
                                              u_label_batch.cuda()

                u_loss = None
                b_loss = None
                if not flag_pair:
                    u_list, perm_list = net_parallel(volume_batch, u_label_batch)
                    u_loss = unary_loss(u_list, perm_list)
                    optimizer.zero_grad()
                    u_loss.backward()

                    print('iteration %d | u_loss %f' % (iter_num, u_loss.item()))

                else:
                    b_label_batch = sampled_batch['b_label']
                    b_label_batch = b_label_batch.cuda()

                    u_list, perm_list, b_stack = net_parallel(volume_batch, 
                                                              u_label_batch)
                    u_loss = unary_loss(u_list, perm_list)
                    b_loss = binary_loss(b_stack, b_label_batch)

                    optimizer.zero_grad()
                    u_loss.backward(retain_graph=True)
                    b_loss.backward()

                    print('iteration %d | u_loss %f | b_loss %f' % \
                          (iter_num, u_loss.item(), b_loss.item()))

                optimizer.step()

                iter_num = iter_num + 1

                if iter_num % 5000 == 0:
                    torch.save(net.state_dict(), 
                               os.path.join(snapshot_path, 
                               snapshot_prefix + '_iter_' + str(iter_num) + '.pth'))

                if iter_num >= max_iterations:
                    break

            if iter_num >= max_iterations:
                break

    if sys.argv[2] == 'test':
        load_snapshot_path = sys.argv[3]
        net.load_state_dict(torch.load(load_snapshot_path))

        print("#############################################")
        dataset = NIHDataset(list_file = test_list_path,
                                root_dir = train_data_path,
                                transform = transforms.Compose([
                                RandomCrop(cell_size, patch_size, puzzle_config),
                                ToTensor(),
                                ]))
        dataloader = DataLoader(dataset, batch_size = 1,
                                shuffle=True, num_workers = 4)

        net.eval()

        while True:
            for i_batch, sampled_batch in enumerate(dataloader):
                volume_batch, label_batch = \
                    sampled_batch['image'], sampled_batch['label']
                volume_batch, label_batch = \
                    volume_batch.cuda(), label_batch.cuda()

                output_list, hungarian_list = net(volume_batch, label_batch)
                loss = puzzle_loss(output_list, hungarian_list)
                print(hungarian_list)

    else:
        print("No Such Arguments!")
