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
		volume2 = volume2.view((-1,) + volume2.shape[2:])
		with torch.cuda.amp.autocast(): 
			q = model(volume, pretrain=True)
			with torch.no_grad():
				k = model_ema(volume2, pretrain=True)

			output = contrast(q, k, all_k=None)
			losses, accuracies = compute_loss_accuracy(
							logits=output[:-1], target=output[-1],
							criterion=criterion)
		
		label  = batch['label'].cuda(args.local_rank)
		label  = label.view((-1,) + label.shape[2:])
		output, _ = model_ema(volume)
		loss = cross_entropy_3d(output, label)
		loss += losses[0] * 0.1

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		pred = output.argmax(dim = 1)
		label = label.squeeze(1)
		d = dice(pred.cpu().data.numpy() == 1, label.cpu().data.numpy() == 1)
		dices.append(d)
		losses.append(loss)
		losses.append(loss.detach().cpu().item())
  
		momentum_update(model, model_ema)
		if i % print_freq == 0 and args.local_rank == 0:
			tqdm.write('[Epoch {}, {}/{}] loss: {}, dice: {}'.format(epoch, i, len(loader), loss.detach().cpu().item(), d))
			logger.log("train/loss", loss)
			logger.log("train/dice", d)
			logger.step()
   
   
	tqdm.write("[Epoch {}] avg loss: {}, avg dice: {}".format(epoch, sum(losses) / len(losses), sum(dices) / len(dices)))
 
def validate(model, loader, optimizer, logger, saver, args, epoch):
	model.eval()
	dices_ = []
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
		dices_.append(d)
		if args.local_rank == 0:
			logger.log("test/dice", d)
			saver.save(epoch, {
					'state_dict': model.state_dict(),
					'dice': d,
					'optimizer_state_dict': optimizer.state_dict()
				}, d)
	tqdm.write("[Epoch {}] test avg dice: {}".format(epoch, sum(dices_) / len(dices_)))


def momentum_update(model, model_ema, m = 0.999):
    """ model_ema = m * model_ema + (1 - m) model """
    for p1, p2 in zip(model.parameters(), model_ema.parameters()):
        p2.data.mul_(m).add_(1 - m, p1.detach().data)