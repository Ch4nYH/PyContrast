import os
import time
import torch
import dateutil.tz

from tqdm import tqdm
from utils import dice, Logger, Saver, adjust_learning_rate
from config import parse_args
from datetime import datetime
from pretrain_functions import pretrain, momentum_update
from datasets.paths import get_paths
from datasets.hdf5 import HDF5Dataset
from datasets.dataset import build_dataset

from torch.utils.data import DataLoader
from models.vnet import VNet
from models.mem_moco import RGBMoCo

import torch.utils.data.distributed
import torch.multiprocessing as mp
from apex.parallel import DistributedDataParallel as DDP
try:
	from apex import amp, optimizers
except ImportError:
	pass

args = parse_args()
args.pretrain = True

os.environ['MASTER_PORT'] = args.port
torch.cuda.set_device(args.local_rank)
torch.distributed.init_process_group(
	'nccl',
    init_method='env://'
)
device = torch.device('cuda:{}'.format(args.local_rank))
now = datetime.now(dateutil.tz.tzlocal())
timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
root_path = 'exps/exp{}_{}'.format(args.exp, timestamp)
if not os.path.exists(root_path):
	os.mkdir(root_path)
	os.mkdir(os.path.join(root_path, "log"))
	os.mkdir(os.path.join(root_path, "model"))

base_lr = args.lr  # base learning rate
batch_size = 1
max_iterations = 40000
cell_size = 96  # size of volume we crop patch from
patch_size = 64
puzzle_config = 3  # 2 or 3 for 2X2X2 or 3X3X3 puzzle
puzzle_num = puzzle_config ** 3
feature_len = 256  #
iter_num = 0
sr_feature_size = 32

train_dataset, val_dataset = build_dataset(args)
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, 
    shuffle=False,
    sampler = train_sampler,
    num_workers=args.num_workers, pin_memory=True)
    
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=1, 
    num_workers=args.num_workers, pin_memory=True)
model = VNet(args.n_channels, args.n_classes, pretrain = True).cuda(args.local_rank)
model_ema = VNet(args.n_channels, args.n_classes, pretrain = True).cuda(args.local_rank)


optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, momentum=0.9, weight_decay=0.0005)
#scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.7)
if args.amp:
	model, optimizer = amp.initialize(
		model, optimizer, opt_level=args.opt_level
	)
	model_ema = amp.initialize(
		model_ema, opt_level=args.opt_level
	)
	model_ema.load_state_dict(model.state_dict())
model = DDP(model)
model_ema = DDP(model_ema)
logger = Logger(root_path)
saver = Saver(root_path, save_freq = args.save_freq)
contrast = RGBMoCo(128, K = 4096).cuda()
criterion = torch.nn.CrossEntropyLoss()
for epoch in tqdm(range(args.start_epoch, args.epochs)):
	train_sampler.set_epoch(epoch)
	pretrain(model, model_ema, train_loader, optimizer, logger, saver, args, epoch, contrast, criterion, args.local_rank)
	adjust_learning_rate(args, optimizer, epoch)
