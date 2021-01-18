import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
import os
import tensorboardX

from losses import TripletLoss
from datasets import TripletMNIST, TripletVeriDataset
from networks import TripletNet, EmbeddingNet, MobileNetv2, Net
from metrics import AccumulatedAccuracyMetric
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torchvision import datasets, transforms
from opts import parse_opts


def fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, device, opt, metrics=[]):
	"""
	Loaders, model, loss function and metrics should work together for a given task,
	i.e. The model should be able to process data output of loaders,
	loss function should process target output of loaders and outputs from the model

	Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
	Siamese network: Siamese loader, siamese model, contrastive loss
	Online triplet learning: batch loader, embedding model, online triplet loss
	"""
	# for epoch in range(0, start_epoch):
	# 	scheduler.step()
	# tensorboard
	summary_writer = tensorboardX.SummaryWriter(log_dir='tf_logs')
	th = 10000
	for epoch in range(opt.start_epoch, opt.n_epochs + 1):
		# Train stage
		train_loss, metrics = train_epoch(
			train_loader, model, loss_fn, optimizer, device, opt, metrics)


		message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(
			epoch, opt.n_epochs, train_loss)
		for metric in metrics:
			message += '\t{}: {}'.format(metric.name(), metric.value())

		val_loss, metrics = test_epoch(
			val_loader, model, loss_fn, device, opt, metrics)
		val_loss /= len(val_loader)

		scheduler.step(val_loss)
		lr = optimizer.param_groups[0]['lr']

		message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch, opt.n_epochs,
																				 val_loss)
		for metric in metrics:
			message += '\t{}: {}'.format(metric.name(), metric.value())

		print(message)

		# if epoch % opt.save_interval == 0:
		if val_loss < th:
			state = {'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
						'optimizer_state_dict': optimizer.state_dict()}
			torch.save(state, os.path.join('snapshots', f'car_re_id_model.pth'))
			print("Epoch {} model saved!\n".format(epoch))
			th = val_loss

		# write summary
		summary_writer.add_scalar(
			'losses/train_loss', train_loss, global_step=epoch)
		summary_writer.add_scalar(
			'losses/val_loss', val_loss, global_step=epoch)
		summary_writer.add_scalar(
			'lr', lr, global_step=epoch)
		# summary_writer.add_scalar(
		# 	'acc/train_acc', train_acc * 100, global_step=epoch)
		# summary_writer.add_scalar(
		# 	'acc/val_acc', val_acc * 100, global_step=epoch)


def train_epoch(train_loader, model, loss_fn, optimizer, device, opt, metrics):
	for metric in metrics:
		metric.reset()

	model.train()
	losses = []
	total_loss = 0

	for batch_idx, (data, target) in enumerate(train_loader):
		target = target if len(target) > 0 else None
		if not type(data) in (tuple, list):
			data = (data,)
		if opt.use_cuda:
			data = tuple(d.to(device) for d in data)
			if target is not None:
				target = target.to(device)

		optimizer.zero_grad()
		outputs = model(*data)

		if type(outputs) not in (tuple, list):
			outputs = (outputs,)

		loss_inputs = outputs
		if target is not None:
			target = (target,)
			loss_inputs += target

		loss_outputs = loss_fn(*loss_inputs)
		loss = loss_outputs[0] if type(loss_outputs) in (
			tuple, list) else loss_outputs
		losses.append(loss.item())
		total_loss += loss.item()
		loss.backward()
		optimizer.step()

		for metric in metrics:
			metric(outputs, target, loss_outputs)

		if (batch_idx + 1) % opt.log_interval == 0:
			message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				(batch_idx + 1) * len(data[0]), len(train_loader.dataset),
				100. * (batch_idx + 1) / len(train_loader), np.mean(losses))
			for metric in metrics:
				message += '\t{}: {}'.format(metric.name(), metric.value())

			print(message)
			losses = []

	total_loss /= (batch_idx + 1)
	return total_loss, metrics


def test_epoch(val_loader, model, loss_fn, device, opt, metrics):
	with torch.no_grad():
		for metric in metrics:
			metric.reset()
		model.eval()
		val_loss = 0
		for batch_idx, (data, target) in enumerate(val_loader):
			target = target if len(target) > 0 else None
			if not type(data) in (tuple, list):
				data = (data,)
			if opt.use_cuda:
				data = tuple(d.to(device) for d in data)
				if target is not None:
					target = target.to(device)

			outputs = model(*data)

			if type(outputs) not in (tuple, list):
				outputs = (outputs,)
			loss_inputs = outputs
			if target is not None:
				target = (target,)
				loss_inputs += target

			loss_outputs = loss_fn(*loss_inputs)
			loss = loss_outputs[0] if type(loss_outputs) in (
				tuple, list) else loss_outputs
			val_loss += loss.item()

			for metric in metrics:
				metric(outputs, target, loss_outputs)

	return val_loss, metrics


if (__name__ == '__main__'):
	torch.manual_seed(1)
	np.random.seed(1)
	# torch.backends.cudnn.deterministic = True
	# torch.backends.cudnn.benchmark = False

	opt = parse_opts()

	# CUDA for PyTorch

	device = torch.device(f"cuda:{opt.gpu}" if opt.use_cuda else "cpu")
	# use_cuda = torch.cuda.is_available()
	# kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

	# training_data = datasets.MNIST('./data', train=True, download=True,
	# 					   transform=transforms.Compose([
	# 						   transforms.ToTensor(),
	# 						   transforms.Normalize((0.1307,), (0.3081,))
	# 					   ]))
	# validation_data =  datasets.MNIST('./data', train=False, transform=transforms.Compose([
	# 						transforms.ToTensor(),
	# 						transforms.Normalize((0.1307,), (0.3081,))
	# 					]))

	train_transform = transforms.Compose([
		transforms.Resize((224, 224)),
		#    transforms.RandomRotation([-45,45]),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
			0.229, 0.224, 0.225])
	])

	test_transform = transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
			0.229, 0.224, 0.225])
	])

	training_data = TripletVeriDataset(
		root_dir=opt.train_images, xml_path=opt.train_annotation_path, transform=train_transform)
	validation_data = TripletVeriDataset(
		root_dir=opt.test_images, xml_path=opt.test_annotation_path, transform=test_transform)

	train_loader = torch.utils.data.DataLoader(training_data,
											   batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)

	val_loader = torch.utils.data.DataLoader(validation_data,
											 batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)

	embedding_net = Net()
	# embedding_net = MobileNetv2()
	model = TripletNet(embedding_net).to(device)
	loss_fn = nn.TripletMarginLoss(margin=0.5)
	# loss_fn = TripletLoss(0.5)

	# optimizer = optim.Adadelta(
	# 	model.parameters(), lr=opt.learning_rate, weight_decay=5e-4)
	optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
	# scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
	scheduler = ReduceLROnPlateau(
			optimizer, 'min', patience=5)
	

	if opt.resume_path:
		print('loading checkpoint {}'.format(opt.resume_path))
		checkpoint = torch.load(opt.resume_path)
		opt.start_epoch = checkpoint['epoch']
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

	fit(train_loader, val_loader, model, loss_fn,
		optimizer, scheduler, device, opt, metrics=[])
