import os
import time

import torch
import dateutil.tz
import pickle
from tqdm import tqdm
from utils.utils import dice, Logger, Saver, adjust_learning_rate
from config import parse_args
from datetime import datetime
from functions import train, validate
from datasets.paths import get_paths
from datasets.hdf5 import HDF5Dataset
from datasets.dataset import build_dataset

from torch.utils.data import DataLoader
from models.vnet_parallel import VNet
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
def main():

    args = parse_args()
    args.pretrain = True
    print("Using GPU: {}".format(args.local_rank))
    
    base_lr = args.lr  # base learning rate
    batch_size = 1
    max_iterations = 20000

    cell_size = 96  # size of volume we crop patch from
    patch_size = 64
    puzzle_config = 3  # 2 or 3 for 2X2X2 or 3X3X3 puzzle
    puzzle_num = puzzle_config ** 3
    feature_len = 256  #
    iter_num = 0
    sr_feature_size = 32
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    train_dataset, val_dataset = build_dataset(args)
    args.world_size = len(args.gpu.split(","))
    if args.world_size > 1:
        os.environ['MASTER_PORT'] = args.port
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            'nccl'
        )
        device = torch.device('cuda:{}'.format(args.local_rank))
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas = len(args.gpu.split(",")), rank = args.local_rank)
    else:
        train_sampler = None
  
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, 
        shuffle=(train_sampler is None),
        sampler = train_sampler,
        num_workers=args.num_workers, pin_memory=True)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, 
        num_workers=args.num_workers, pin_memory=True)

    model = VNet(args.n_channels, args.n_classes, input_size = 64, pretrain = True).cuda(args.local_rank)
    model_ema = VNet(args.n_channels, args.n_classes, input_size = 64, pretrain = True).cuda(args.local_rank)
    optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, momentum=0.9, weight_decay=0.0005)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.7)
    parallel_state_dict = torch.load(args.load_path)['state_dict']
    new_state_dict = {}
    for key in parallel_state_dict.keys():
        new_state_dict[key[7:]] = parallel_state_dict[key]

    model.load_state_dict(new_state_dict)
    model.eval()
    print("Loaded weights")
    print("Using Dataset: {}".format(type(train_dataset)))

    features = []
    for i, batch in enumerate(tqdm(train_loader)):
        volume = batch['image'].cuda(args.local_rank, non_blocking = True)
        volume = volume.view((-1,) + volume.shape[2:])

        with torch.no_grad():
            q = model(volume, pretrain=True)

        features.append(q)
        if i > 99:
            break
    features = torch.cat(features, 0)

    pickle.dump(features.cpu().numpy(), open("features.pkl", 'wb'))

    
if __name__ == "__main__":
    main()