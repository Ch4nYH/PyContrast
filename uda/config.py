import argparse

def parse_args():
	parser = argparse.ArgumentParser(description='Training.')
	parser.add_argument('-d', '--dataset', nargs='+', default=[], help='The datasets to be trained')
	parser.add_argument('--exp', type=str, help = "Name of experiments")
	parser.add_argument('--lr', type=float, default = 1e-2)
	parser.add_argument('--seed', type=int, default = 42)
	parser.add_argument('--start-epoch', type=int, default=0)
	parser.add_argument('--epochs', type=int, default=100)
	parser.add_argument('--batch-size', type=int, default=1)
	parser.add_argument('--num-workers', type=int, default=2)
	parser.add_argument('--n-classes', type=int, default=2)
	parser.add_argument('--n-channels', type=int, default=1)
	parser.add_argument('--print-freq', type=int, default=10)
	parser.add_argument('--save-freq', type=int, default=100)
	parser.add_argument('--amp', action="store_true")
	parser.add_argument('--opt_level', type=str, default='O1',
							choices=['O1', 'O2'])

	parser.add_argument('--load-path', type=str, default=None)
	parser.add_argument('--world-size', default=-1, type=int,
						help='number of nodes for distributed training')
	parser.add_argument('--rank', default=-1, type=int,
						help='node rank for distributed training')
	parser.add_argument('--dist-url', default='tcp://127.0.0.1:23456', type=str,
						help='url used to set up distributed training')
	parser.add_argument('--dist-backend', default='nccl', type=str,
						help='distributed backend')
	parser.add_argument('--gpu', default=None, type=str,
						help='GPU id to use.')
	parser.add_argument('--multiprocessing-distributed', action='store_true',
							help='Use multi-processing distributed training to launch '
								 'N processes per node, which has N GPUs. This is the '
								 'fastest way to use PyTorch for either single node or '
								 'multi node data parallel training')
	parser.add_argument("--local_rank", type=int, default=0)
	parser.add_argument("--port", type=str, default="15000")
	parser.add_argument("--data-root", default="/ccvl/net/ccvl15/shuhao/domain_adaptation/datasets")
	parser.add_argument("--train-list", default=None, type=str)
	parser.add_argument("--ssim", action="store_true", default=False)	
 
	parser.add_argument("--resume", type=str, help="resume checkpoint")
 
	parser.add_argument('--coef', type=float, default=0.2)
	parser.add_argument('--increasing-coef', action="store_true")
	parser.add_argument('--turnon', default=100, type=int)
	parser.add_argument('--memory', default='default', type=str, choices=['default', '4layers'])
	parser.add_argument('--temperature', default=0.07, type=float)
	args = parser.parse_args()
	return args
