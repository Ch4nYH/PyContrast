import os
import time
import torch
import dateutil.tz

from tqdm import tqdm
from utils.utils import dice, Logger, Saver, adjust_learning_rate
from config import parse_args
from datetime import datetime
from pretrain_functions import pretrain, momentum_update
from datasets import build_dataset

from torch.utils.data import DataLoader
from models.vnet import VNet
from models.mem_moco import RGBMoCo, RGBMoCoNew

import torch.utils.data.distributed
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
apex = False

args = parse_args()
args.pretrain = True

root_path = 'exps/exp{}'.format(args.exp)
if not os.path.exists(root_path):
	os.mkdir(root_path)
	os.mkdir(os.path.join(root_path, "log"))
	os.mkdir(os.path.join(root_path, "model"))

base_lr = args.lr  # base learning rate
train_dataset, val_dataset = build_dataset(args.dataset, args.data_root, args.train_list, sampling = args.sampling)
    
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, 
    shuffle=False,
    num_workers=args.num_workers, pin_memory=True, drop_last = True)
    
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=1, 
    num_workers=args.num_workers, pin_memory=True)
model = VNet(args.n_channels, args.n_classes, input_size = 64, pretrain = True).cuda()
model_ema = VNet(args.n_channels, args.n_classes, input_size = 64, pretrain = True).cuda()


optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, momentum=0.9, weight_decay=0.0005)
model = torch.nn.DataParallel(model)
model_ema = torch.nn.DataParallel(model_ema)
model_ema.load_state_dict(model.state_dict())
print("Model Initialized")
logger = Logger(root_path)
saver = Saver(root_path, save_freq = args.save_freq)
if args.sampling == 'default':
    contrast = RGBMoCo(128, K = 4096, T = args.temperature).cuda()
elif args.sampling == 'layerwise':
    contrast = RGBMoCoNew(128, K = 4096, T = args.temperature).cuda()
else:
    raise ValueError("unsupported sampling method")
criterion = torch.nn.CrossEntropyLoss()
for epoch in range(args.start_epoch, args.epochs):
	pretrain(model, model_ema, train_loader, optimizer, logger, saver, args, epoch, contrast, criterion)
	adjust_learning_rate(args, optimizer, epoch)
