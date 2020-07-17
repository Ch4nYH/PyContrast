import argparse

def parse_args():
	parser = argparse.ArgumentParser(description='Training.')
	parser.add_argument('--dataset', type=str, default='', help='The dataset to be trained')
	parser.add_argument('-g', '--gpu', default = None, help = "gpu")
	parser.add_argument('--exp', type=str, help = "Name of experiments")
	parser.add_argument('--lr', type=float, default = 1e-2)
	parser.add_argument('--seed', type=int, default = 42)
	parser.add_argument('--start-epoch', type=int, default=0)
	parser.add_argument('--epochs', type=int, default=100)
	parser.add_argument('--batch-size', type=int, default=1)
	parser.add_argument('--num-workers', type=int, default=8)
	parser.add_argument('--n-classes', type=int, default=2)
	parser.add_argument('--n-channels', type=int, default=1)
	parser.add_argument('--print-freq', type=int, default=10)
	parser.add_argument('--amp', action="store_true")
	parser.add_argument('--opt_level', type=str, default='O2',
                            choices=['O1', 'O2'])

	args = parser.parse_args()

	return args
