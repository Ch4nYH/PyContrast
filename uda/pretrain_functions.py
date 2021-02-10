import torch
import torch.nn as nn
from tqdm import tqdm
from utils.utils import cross_entropy_3d, dice, AverageMeter, compute_loss_accuracy
from models import mem_moco

def pretrain(model, model_ema, loader, optimizer, logger, saver, args, epoch, contrast, criterion):
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
	scaler = torch.cuda.amp.GradScaler() 
	for i, batch in enumerate(tqdm(loader)):
		index = batch['index']
		volume = batch['image'].cuda(args.local_rank, non_blocking = True)
		volume = volume.view((-1,) + volume.shape[2:])

		volume2 = batch['image_2'].cuda(args.local_rank, non_blocking = True)
		volume2 = volume2.view((-1,) + volume2.shape[2:])
  
		with torch.cuda.amp.autocast(): 
			q = model(volume)
			with torch.no_grad():
				k = model_ema(volume2)

			output = contrast(q, k, all_k=None)
			losses, accuracies = compute_loss_accuracy(
							logits=output[:-1], target=output[-1],
							criterion=criterion)
		loss = losses[0]
		update_loss = losses[0]
		update_acc = accuracies[0]
		update_loss_jig = torch.tensor([0.0])
		update_acc_jig = torch.tensor([0.0])

		#loss = cross_entropy_3d(output, label)

		optimizer.zero_grad()
		scaler.scale(loss).backward() 
		scaler.step(optimizer)
		scaler.update() 

		loss_meter.update(update_loss.item(), args.batch_size)
		loss_jig_meter.update(update_loss_jig.item(), args.batch_size)
		acc_meter.update(update_acc[0], args.batch_size)
		acc_jig_meter.update(update_acc_jig[0], args.batch_size)
		momentum_update(model, model_ema)
		if i % args.print_freq == 0 and args.local_rank == 0:
			tqdm.write('Train: [{0}][{1}/{2}]\t'
						  'l_I {loss.val:.3f} ({loss.avg:.3f})\t'
						  'a_I {acc.val:.3f} ({acc.avg:.3f})\t'
						  'l_J {loss_jig.val:.3f} ({loss_jig.avg:.3f})\t'
						  'a_J {acc_jig.val:.3f} ({acc_jig.avg:.3f})'.format(
						   epoch, i + 1, len(loader), loss=loss_meter, acc=acc_meter,
						   loss_jig=loss_jig_meter, acc_jig=acc_jig_meter))
		if args.local_rank == 0:
			logger.log("pretrain/loss", update_loss.item())
			logger.log("pretrain/acc", update_acc[0])
			logger.step()
	if args.local_rank == 0:
		saver.save(epoch, {
			'state_dict': model_ema.state_dict(),
			'acc': acc_meter.avg,
			'optimizer_state_dict': optimizer.state_dict(),
			#'amp': amp.state_dict() if args.amp else None
		}, acc_meter.avg)

def momentum_update(model, model_ema, m = 0.999):
        """ model_ema = m * model_ema + (1 - m) model """
        for p1, p2 in zip(model.parameters(), model_ema.parameters()):
            p2.data.mul_(m).add_(1 - m, p1.detach().data)

def unary_loss(output_list, label_list):
	loss_list = []
	puzzle_num = output_list[0].shape[2]
	loss = nn.NLLLoss()

	for i in range(len(output_list)):
		output = output_list[i].view(-1, puzzle_num)
		label = label_list[i].view(-1)
		loss_list.append(loss(output, label))
	return torch.mean(torch.stack(loss_list))

def binary_loss(b_stack, b_label_batch):
	b_label_batch = b_label_batch.view(-1)
	loss = nn.NLLLoss()

	return loss(b_stack, b_label_batch)


def pretrain_jigsaw(model, model_ema, loader, optimizer, logger, saver, args, epoch, contrast, criterion):
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
	scaler = torch.cuda.amp.GradScaler() 
	for i, batch in enumerate(tqdm(loader)):
		index = batch['index']
		volume = batch['image'].cuda(non_blocking = True)
		volume = volume.view((-1, 8,) + volume.shape[3:])

		volume2 = batch['image_2'].cuda(non_blocking = True)
		volume2 = volume2.view((-1, 8,) + volume2.shape[3:])
		print(volume.shape)
		with torch.cuda.amp.autocast(): 
			_, q, unary_list1, perm_list1, binary_stack1 = model(volume, batch['u_label'], batch['b_label'])
			with torch.no_grad():
				_, k, unary_list2, perm_list2, binary_stack2 = model_ema(volume2, batch['u_label_2'], batch['b_label_2'])
			
			k = k[batch['u_label_2'].view(-1).long()]
			q = q[batch['u_label'].view(-1).long()]

			u_loss = unary_loss(unary_list1, perm_list1)
			b_loss = binary_loss(binary_stack1, batch['b_label'])
			output = contrast(q, k, all_k=None)
			losses, accuracies = compute_loss_accuracy(
							logits=output[:-1], target=output[-1],
							criterion=criterion)
		loss = losses[0]
		update_loss = losses[0]
		update_acc = accuracies[0]
		update_loss_jig = torch.tensor([0.0])
		update_acc_jig = torch.tensor([0.0])

		#loss = cross_entropy_3d(output, label)

		optimizer.zero_grad()
		scaler.scale(loss + u_loss + b_loss).backward() 
		scaler.step(optimizer)
		scaler.update() 

		loss_meter.update(update_loss.item(), args.batch_size)
		loss_jig_meter.update(update_loss_jig.item(), args.batch_size)
		acc_meter.update(update_acc[0], args.batch_size)
		acc_jig_meter.update(update_acc_jig[0], args.batch_size)
		momentum_update(model, model_ema)
		if i % args.print_freq == 0 and args.local_rank == 0:
			tqdm.write('Train: [{0}][{1}/{2}]\t'
						  'l_I {loss.val:.3f} ({loss.avg:.3f})\t'
						  'a_I {acc.val:.3f} ({acc.avg:.3f})\t'
						  'l_J {loss_jig.val:.3f} ({loss_jig.avg:.3f})\t'
						  'a_J {acc_jig.val:.3f} ({acc_jig.avg:.3f})'.format(
						   epoch, i + 1, len(loader), loss=loss_meter, acc=acc_meter,
						   loss_jig=loss_jig_meter, acc_jig=acc_jig_meter))
		if args.local_rank == 0:
			logger.log("pretrain/loss", update_loss.item())
			logger.log("pretrain/acc", update_acc[0])
			logger.step()
	if args.local_rank == 0:
		saver.save(epoch, {
			'state_dict': model_ema.state_dict(),
			'acc': acc_meter.avg,
			'optimizer_state_dict': optimizer.state_dict(),
			#'amp': amp.state_dict() if args.amp else None
		}, acc_meter.avg)