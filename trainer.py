import torch
import numpy as np
from losses import TripletLoss
from datasets import TripletMNIST, TripletVeriDataset
from networks import TripletNet, EmbeddingNet, Resnet18
from metrics import AccumulatedAccuracyMetric
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
import torch.optim as optim
import os
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

	for epoch in range(opt.start_epoch, opt.n_epochs + 1):

		# Train stage
		train_loss, metrics = train_epoch(
			train_loader, model, loss_fn, optimizer, device, opt, metrics)

		scheduler.step()

		message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(
			epoch, opt.n_epochs, train_loss)
		for metric in metrics:
			message += '\t{}: {}'.format(metric.name(), metric.value())

		val_loss, metrics = test_epoch(val_loader, model, loss_fn, device, opt, metrics)
		val_loss /= len(val_loader)

		message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch, opt.n_epochs,
																				 val_loss)
		for metric in metrics:
			message += '\t{}: {}'.format(metric.name(), metric.value())

		print(message)


		if epoch % opt.save_interval == 0:
			state = {'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
					'optimizer_state_dict': optimizer.state_dict()}
			torch.save(state, os.path.join('snapshots', f'model{epoch}.pth'))
			print("Epoch {} model saved!\n".format(epoch))


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

		if batch_idx % log_interval == 0:
			message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				batch_idx * len(data[0]), len(train_loader.dataset),
				100. * batch_idx / len(train_loader), np.mean(losses))
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

opt = parse_opts()

# CUDA for PyTorch

device = torch.device(f"cuda:{opt.gpu}" if opt.cuda else "cpu")
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

transform = transforms.Compose([
						   transforms.Resize((150,150)),
						   transforms.ToTensor(),
						   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
					0.229, 0.224, 0.225])
					   ])

training_data = TripletVeriDataset(root_dir=opt.train_images, xml_path=opt.train_label, transform=transform)
validation_data = TripletVeriDataset(root_dir=opt.test_images, xml_path=opt.test_label, transform=transform)


train_loader=torch.utils.data.DataLoader(training_data,
		batch_size = opt.batch_size, shuffle = True, num_workers=opt.num_workers)

val_loader=torch.utils.data.DataLoader(validation_data,
		batch_size = opt.batch_size, shuffle = True, num_workers=opt.num_workers)


embedding_net=Resnet18()
model=TripletNet(embedding_net).to(device)
loss_fn=TripletLoss(0.5)

optimizer=optim.Adadelta(model.parameters(), lr = 0.1)

scheduler=StepLR(optimizer, step_size = 1, gamma = 0.1)


fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, device, opt, metrics = [])
