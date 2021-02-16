import os
import time
import torch
import dateutil.tz

from tqdm import tqdm
from utils.utils  import dice, Logger, Saver, adjust_learning_rate
from config import parse_args
from datetime import datetime
from functions import train, validate
from datasets.paths import get_paths
from datasets.hdf5 import HDF5Dataset
from datasets import build_dataset

from torch.utils.data import DataLoader
from models.vnet import VNet
from torch.nn.parallel import DistributedDataParallel as DDP

def main():

    args = parse_args()
    args.pretrain = False
    
    root_path = 'exps/exp_{}'.format(args.exp)
 
    if not os.path.exists(root_path):
        os.mkdir(root_path)
        os.mkdir(os.path.join(root_path, "log"))
        os.mkdir(os.path.join(root_path, "model"))
    
    base_lr = args.lr  # base learning rate
    
    train_dataset, val_dataset = build_dataset(args.dataset, args.data_root, args.train_list)
  
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers, pin_memory=True)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, 
        num_workers=args.num_workers, pin_memory=True)

    model = VNet(args.n_channels, args.n_classes).cuda()


    
    optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, momentum=0.9, weight_decay=0.0005)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.7)
    
    model = torch.nn.DataParallel(model)

    model.train()
    
    if args.resume is None:
        assert os.path.exists(args.load_path)
        state_dict = model.state_dict()
        print("Loading weights...")
        pretrain_state_dict = torch.load(args.load_path, map_location="cpu")['state_dict']
        
        for k in list(pretrain_state_dict.keys()):
            if k not in state_dict:
                del pretrain_state_dict[k]
        model.load_state_dict(pretrain_state_dict)
        print("Loaded weights")
    else:
        print("Resuming from {}".format(args.resume))
        checkpoint = torch.load(args.resume, map_location="cpu")
        
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.load_state_dict(checkpoint['state_dict'])
    
    

    logger = Logger(root_path)
    saver = Saver(root_path)

    for epoch in range(args.start_epoch, args.epochs):
        train(model, train_loader, optimizer, logger, args, epoch)
        validate(model, val_loader, optimizer, logger, saver, args, epoch)
        adjust_learning_rate(args, optimizer, epoch)

if __name__ == '__main__':
    main()
