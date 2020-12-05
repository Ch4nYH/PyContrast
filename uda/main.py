import os
import time
import torch
import dateutil.tz

from tqdm import tqdm
from utils.utils import dice, Logger, Saver, adjust_learning_rate
from config import parse_args
from datetime import datetime
from functions import train, validate
from datasets import build_dataset

from torch.utils.data import DataLoader
from models.vnet import VNet
from torch.nn.parallel import DistributedDataParallel as DDP

def main():

    args = parse_args()
    args.pretrain = False
    print("Using GPU: {}".format(args.local_rank))
    root_path = 'exps/exp_{}'.format(args.exp)
    if args.local_rank == 0 and not os.path.exists(root_path):
        os.mkdir(root_path)
        os.mkdir(os.path.join(root_path, "log"))
        os.mkdir(os.path.join(root_path, "model"))
    
    base_lr = args.lr  # base learning rate
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    train_dataset, val_dataset = build_dataset(args.dataset, args.data_root, args.train_list)
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

    model = VNet(args.n_channels, args.n_classes).cuda(args.local_rank)
    
    optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, momentum=0.9, weight_decay=0.0005)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.7)
    if args.world_size > 1:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    model.train()
    print("Loaded weights")

    logger = Logger(root_path)
    saver = Saver(root_path)

    for epoch in range(args.start_epoch, args.epochs):
        train(model, train_loader, optimizer, logger, args, epoch)
        validate(model, val_loader, optimizer, logger, saver, args, epoch)
        adjust_learning_rate(args, optimizer, epoch)


if __name__ == '__main__':
    main()
