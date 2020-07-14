import torch

from tqdm import tqdm
from utils import cross_entropy_3d, dice

def train(model, loader, optimizer, scheduler, logger, args, epoch):
	model.train()
	for i, batch in enumerate(tqdm(loader)):
		index = batch['index']
		volume = batch['image'].cuda()
		volume = volume.view((-1,) + volume.shape[2:])
		label  = batch['label'].cuda()
		label  = label.view((-1,) + label.shape[2:])

		output = model(volume)
		loss = cross_entropy_3d(output, label)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		pred = output.argmax(dim = 1)
		d = dice(pred.cpu().data.numpy() == 1, label.cpu().data.numpy() == 1)
		if i % 100 == 0:
			print('[Epoch {}, {}/{}] loss: {}'.format(epoch, i, len(loader), loss.detach().cpu().item()))

		logger.log("train/loss", loss)
		logger.log("train/dice", d)
		logger.step()

	scheduler.step()

def validate(model, loader, optimizer, logger, saver, args, epoch):
	model.eval()
	for i, batch in enumerate(tqdm(loader)):
		index = batch['index']
		volume = batch['image'].cuda()
		volume = volume.view((-1,) + volume.shape[2:])
		label  = batch['label'].cuda()
		label  = label.view((-1,) + label.shape[2:])

		output = model(volume)
		pred = output.argmax(dim = 1)
		d = dice(pred.cpu().data.numpy() == 1, label.cpu().data.numpy() == 1)
		logger.log("train/dice", d)
		saver.save(epoch, {
				'state_dict': model.state_dict(),
				'dice': dice,
				'optimizer_state_dict': optimizer.state_dict()
			}, dice)

