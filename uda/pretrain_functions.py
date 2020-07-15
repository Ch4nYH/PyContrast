import torch

from tqdm import tqdm
from utils import cross_entropy_3d, dice, AverageMeter
from models import mem_moco

def validate(model, loader, optimizer, logger, saver, args, epoch):
	model.eval()
	dices = []
	for i, batch in enumerate(loader):
		index = batch['index']
		volume = batch['image'].cuda()
		volume = volume.view((-1,) + volume.shape[2:])
		label  = batch['label'].cuda()
		label  = label.view((-1,) + label.shape[2:])

		output = model(volume)
		pred = output.argmax(dim = 1)
		d = dice(pred.cpu().data.numpy() == 1, label.cpu().data.numpy() == 1)
		logger.log("train/dice", d)
		dices.append(d)
		saver.save(epoch, {
				'state_dict': model.state_dict(),
				'dice': d,
				'optimizer_state_dict': optimizer.state_dict()
			}, d)
	tqdm.write("[Epoch {}] test avg dice: {}".format(epoch, sum(dices) / len(dices)))



def pretrain(model, model_ema, loader, optimizer, logger, args, epoch, contrast, criterion):
	model.train()
	model_ema.eval()

	batch_time = AverageMeter()
	data_time = AverageMeter()
	loss_meter = AverageMeter()
	acc_meter = AverageMeter()
	loss_jig_meter = AverageMeter()
	acc_jig_meter = AverageMeter()

	def set_bn_train(m):
		classname = m.__class__.__name__
		if classname.find('BatchNorm') != -1:
			m.train()
	model_ema.apply(set_bn_train)

	for i, batch in enumerate(tqdm(loader)):
		index = batch['index']
		volume = batch['image'].cuda()
		volume = volume.view((-1,) + volume.shape[2:])

		volume2 = batch['image_2'].cuda()
		volume2 = volume2.view((-1,) + volume2.shape[2:])

		q = model(volume)
		k = model_ema(volume)

		output = contrast(q, k, all_k=None)
		losses, accuracies = self._compute_loss_accuracy(
						logits=output[:-1], target=output[-1],
						criterion=criterion)
		loss = losses[0]
		update_loss = losses[0]
		update_acc = accuracies[0]
		update_loss_jig = torch.tensor([0.0])
		update_acc_jig = torch.tensor([0.0])

		#loss = cross_entropy_3d(output, label)

		optimizer.zero_grad()
		if args.amp:
			with amp.scale_loss(loss, optimizer) as scaled_loss:
					scaled_loss.backward()
		else:
			loss.backward()
		optimizer.step()

		loss_meter.update(update_loss.item(), args.batch_size)
		loss_jig_meter.update(update_loss_jig.item(), args.batch_size)
		acc_meter.update(update_acc[0], args.batch_size)
		acc_jig_meter.update(update_acc_jig[0], args.batch_size)
		if i % args.print_freq == 0:
			tqdm.write('Train: [{0}][{1}/{2}]\t'
						  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
						  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
						  'l_I {loss.val:.3f} ({loss.avg:.3f})\t'
						  'a_I {acc.val:.3f} ({acc.avg:.3f})\t'
						  'l_J {loss_jig.val:.3f} ({loss_jig.avg:.3f})\t'
						  'a_J {acc_jig.val:.3f} ({acc_jig.avg:.3f})'.format(
						   epoch, idx + 1, len(train_loader), batch_time=batch_time,
						   data_time=data_time, loss=loss_meter, acc=acc_meter,
						   loss_jig=loss_jig_meter, acc_jig=acc_jig_meter))

		logger.log("pretrain/loss", update_loss.item())
		logger.log("pretrain/acc", update_acc[0])
		

		losses.append(loss.detach().cpu().item())
		dices.append(d)
		logger.step()
	tqdm.write("[Epoch {}] avg loss: {}, avg dice: {}".format(epoch, sum(losses) / len(losses), sum(dices) / len(dices)))
