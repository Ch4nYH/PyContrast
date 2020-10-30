import os
import time
import torch
import dateutil.tz

from tqdm import tqdm
from datetime import datetime

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from config import parse_args
from functions import train, validate
from pretrain_functions import pretrain
from utils.utils  import dice, Logger, Saver, adjust_learning_rate
from datasets.paths import get_paths
from datasets.hdf5 import HDF5Dataset
from datasets.dataset import build_dataset

from models.vnet_parallel import VNet
from models.mem_moco import RGBMoCo

from utils.utils import cross_entropy_3d, dice, AverageMeter, compute_loss_accuracy


def main():

	args = parse_args()
	args.pretrain = True
	print("Using GPU: {}".format(args.local_rank))
	
	root_path = 'exps/exp_{}'.format(args.exp)

	if not os.path.exists(root_path) and args.local_rank == 0:
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

	model = VNet(args.n_channels, args.n_classes, input_size = 64, pretrain=True).cuda(args.local_rank)
	model_ema = VNet(args.n_channels, args.n_classes, input_size = 64, pretrain=True).cuda(args.local_rank)
	assert os.path.exists(args.load_path)

	state_dict = model.state_dict()
	print("Loading weights...")
	optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, momentum=0.9, weight_decay=0.0005)
	
	#scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.7)
	if args.world_size > 1:
		model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

	model.train()
	model_ema = DDP(model_ema, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
	

	pretrain_state_dict = torch.load(args.load_path, map_location="cpu")['state_dict']
		
	model.load_state_dict(pretrain_state_dict)
	model_ema.load_state_dict(model.state_dict())
	print("Loaded weights")
	
	logger = Logger(root_path)
	saver = Saver(root_path, save_freq=args.save_freq)
	contrast = RGBMoCo(128, K = 512).cuda(args.local_rank)
	criterion = torch.nn.CrossEntropyLoss()
	for epoch in range(args.start_epoch, args.epochs):
		train_sampler.set_epoch(epoch)
		adapt(model, model_ema, train_loader, optimizer, logger, saver, args, epoch, contrast, criterion)
		adjust_learning_rate(args, optimizer, epoch)



def adapt(model, model_ema, loader, optimizer, logger, saver, args, epoch, contrast, criterion, print_freq=1):
	losses = []
	dices = []
	model.train()
	model_ema.eval()
	def set_bn_train(m):
			classname = m.__class__.__name__
			if classname.find('BatchNorm') != -1:
				m.train()
	model_ema.apply(set_bn_train)
	scaler = torch.cuda.amp.GradScaler() 
 
	for i, batch in enumerate(loader):
		index = batch['index']
		volume = batch['image'].cuda(args.local_rank, non_blocking = True)
		volume = volume.view((-1,) + volume.shape[2:])
		volume2 = batch['image_2'].cuda(args.local_rank, non_blocking = True)
		volume2 = volume2.view((-1,) + volume2.shape[2:])
		
		q = model(volume, pretrain=True)
		with torch.no_grad():
			k = model_ema(volume2, pretrain=True)

		output = contrast(q, k, all_k=None)
		losses, accuracies = compute_loss_accuracy(
						logits=output[:-1], target=output[-1],
						criterion=criterion)
		optimizer.zero_grad()
		if not args.increasing_coef:
			(losses[0] * args.coef).backward()
		else:
			(losses[0] * epoch / args.epochs * args.coef).backward()

		optimizer.step()
  
		momentum_update(model, model_ema)
		if i % print_freq == 0 and args.local_rank == 0:
			tqdm.write('[Epoch {}, {}/{}] contrast acc: {}'.format(epoch, i, len(loader), accuracies[0][0]))
	
	saver.save(epoch,  {
			'state_dict': model_ema.state_dict()}, 0)
 
def momentum_update(model, model_ema, m = 0.999):
    """ model_ema = m * model_ema + (1 - m) model """
    for p1, p2 in zip(model.parameters(), model_ema.parameters()):
        p2.data.mul_(m).add_(1 - m, p1.detach().data)

if __name__ == '__main__':
	main()
