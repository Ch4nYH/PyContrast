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

try:
	from apex import amp, optimizers
except ImportError:
	pass

def main():

	args = parse_args()
	os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
	print("Using GPU: {}".format(args.gpu))

	torch.manual_seed(args.seed)
	torch.cuda.manual_seed(args.seed)

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


	logger = Logger(root_path)
	saver = Saver(root_path, save_freq = args.save_freq)
	contrast = RGBMoCo(128, K = 4096).cuda().half()
	criterion = torch.nn.CrossEntropyLoss()
	for epoch in tqdm(range(args.start_epoch, args.epochs)):
		pretrain(model, model_ema, train_loader, optimizer, logger, saver, args, epoch, contrast, criterion)

		adjust_learning_rate(args, optimizer, epoch)


if __name__ == '__main__':
	main()
