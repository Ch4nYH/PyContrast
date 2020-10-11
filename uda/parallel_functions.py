import torch

from tqdm import tqdm
from utils.utils import cross_entropy_3d, dice, AverageMeter, compute_loss_accuracy

def train(model, model_ema, loader, optimizer, logger, saver, args, epoch, contrast, criterion, print_freq=10):
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

		with torch.cuda.amp.autocast(): 
			q = model.pretrain_forward(volume)
			with torch.no_grad():
				k = model_ema.pretrain_forward(volume2)

			output = contrast(q, k, all_k=None)
			losses, accuracies = compute_loss_accuracy(
							logits=output[:-1], target=output[-1],
							criterion=criterion)
		
		label  = batch['label'].cuda(args.local_rank)
		label  = label.view((-1,) + label.shape[2:])
		output, _ = model_ema(volume)
		loss = cross_entropy_3d(output, label)
		loss += losses[0]

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

	model.eval()
	dices = []
def validate(model, loader, optimizer, logger, saver, args, epoch):
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
			logger.log("test/dice", d)
			saver.save(epoch, {
					'state_dict': model.state_dict(),
					'dice': d,
					'optimizer_state_dict': optimizer.state_dict()
				}, d)
	tqdm.write("[Epoch {}] test avg dice: {}".format(epoch, sum(dices) / len(dices)))
