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
from datasets.dataset import build_dataloader

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
gpu = args.gpu
gpu_ids = args.gpu.split(',')
args.gpu = []
for gpu_id in gpu_ids:
	id = int(gpu_id)
	args.gpu.append(id)
print(args.gpu)
if len(args.gpu_ids) > 0:
   torch.cuda.set_device(args.gpu_ids[0])

os.environ['MASTER_PORT'] = args.port
torch.distributed.init_process_group(backend="nccl")

now = datetime.now(dateutil.tz.tzlocal())
timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
root_path = 'exps/exp{}_{}'.format(args.exp, timestamp)
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

train_loader, val_loader = build_dataloader(args)
model = VNet(args.n_channels, args.n_classes, pretrain = True).cuda()
model_ema = VNet(args.n_channels, args.n_classes, pretrain = True).cuda()


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
	pretrain(model, model_ema, train_loader, optimizer, logger, saver, args, epoch, contrast, criterion)
	adjust_learning_rate(args, optimizer, epoch)
