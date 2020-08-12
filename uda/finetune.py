import os
import time
import torch
import dateutil.tz

from tqdm import tqdm
from utils import dice, Logger, Saver, adjust_learning_rate
from config import parse_args
from datetime import datetime
from functions import train, validate
from datasets.paths import get_paths
from datasets.hdf5 import HDF5Dataset
from datasets.dataset import build_dataset

from torch.utils.data import DataLoader
from models.vnet import VNet

from apex.parallel import DistributedDataParallel as DDP
try:
	from apex import amp, optimizers
except ImportError:
	pass
def main():

	args = parse_args()
	args.pretrain = False
	print("Using GPU: {}".format(args.local_rank))
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
	rain_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas = 2, rank = args.local_rank)

	model = VNet(args.n_channels, args.n_classes).cuda()
	
	optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, momentum=0.9, weight_decay=0.0005)
	#scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.7)
 	model, optimizer = amp.initialize(
		model, optimizer, opt_level=args.opt_level
	)
	model = DDP(model)
	assert os.path.exists(args.load_path)

	state_dict = model.state_dict()
	pretrain_state_dict = torch.load(args.load_path)['state_dict']

	for k in list(pretrain_state_dict.keys()):
		if k not in state_dict:
			del pretrain_state_dict[k]

	model.load_state_dict(state_dict)
	model.train()

	logger = Logger(root_path)
	saver = Saver(root_path)

	for epoch in tqdm(range(args.start_epoch, args.epochs)):
		train(model, train_loader, optimizer, logger, args, epoch)
		validate(model, val_loader, optimizer, logger, saver, args, epoch)
		adjust_learning_rate(args, optimizer, epoch)


if __name__ == '__main__':
	main()
