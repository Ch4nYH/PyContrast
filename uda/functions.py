import torch

from tqdm import tqdm
from utils import cross_entropy_3d, dice


def train(model, loader, optimizer, logger, args, epoch, print_freq = 10):
	losses = []
	dices = []
	model.train()
	for i, batch in enumerate(loader):
		index = batch['index']
		volume = batch['image'].cuda(args.local_rank)
		volume = volume.view((-1,) + volume.shape[2:])
		label  = batch['label'].cuda(args.local_rank)
		label  = label.view((-1,) + label.shape[2:])
		output, _ = model(volume)
		loss = cross_entropy_3d(output, label)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		pred = output.argmax(dim = 1)
		label = label.squeeze(1)
		d = dice(pred.cpu().data.numpy() == 1, label.cpu().data.numpy() == 1)
 		dices.append(d)
		losses.append(loss)
		losses.append(loss.detach().cpu().item())
		if i % print_freq == 0 and args.local_rank == 0:
			tqdm.write('[Epoch {}, {}/{}] loss: {}, dice: {}'.format(epoch, i, len(loader), loss.detach().cpu().item(), d))

			logger.log("train/loss", loss)
			logger.log("train/dice", d)
			logger.step()
	tqdm.write("[Epoch {}] avg loss: {}, avg dice: {}".format(epoch, sum(losses) / len(losses), sum(dices) / len(dices)))

def validate(model, loader, optimizer, logger, saver, args, epoch):
	model.eval()
	dices = []
	for i, batch in enumerate(loader):
		index = batch['index']
		volume = batch['image'].cuda()
		volume = volume.view((-1,) + volume.shape[2:])
		label  = batch['label'].cuda()
		label  = label.view((-1,) + label.shape[2:])
		label = label.squeeze(1)
		output, _ = model(volume)
		pred = output.argmax(dim = 1)
		d = dice(pred.cpu().data.numpy() == 1, label.cpu().data.numpy() == 1)
  		dices.append(d)
		if args.local_rank == 0:
			logger.log("train/dice", d)
			saver.save(epoch, {
					'state_dict': model.state_dict(),
					'dice': d,
					'optimizer_state_dict': optimizer.state_dict()
				}, d)
	tqdm.write("[Epoch {}] test avg dice: {}".format(epoch, sum(dices) / len(dices)))
