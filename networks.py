import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
# from resnet import resnet10
from torchvision.models import mobilenet_v2
import torch


class BasicBlock(nn.Module):
	def __init__(self, c_in, c_out,is_downsample=False):
		super(BasicBlock,self).__init__()
		self.is_downsample = is_downsample
		if is_downsample:
			self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=2, padding=1, bias=False)
		else:
			self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=1, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(c_out)
		self.relu = nn.ReLU(True)
		self.conv2 = nn.Conv2d(c_out,c_out,3,stride=1,padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(c_out)
		if is_downsample:
			self.downsample = nn.Sequential(
				nn.Conv2d(c_in, c_out, 1, stride=2, bias=False),
				nn.BatchNorm2d(c_out)
			)
		elif c_in != c_out:
			self.downsample = nn.Sequential(
				nn.Conv2d(c_in, c_out, 1, stride=1, bias=False),
				nn.BatchNorm2d(c_out)
			)
			self.is_downsample = True

	def forward(self,x):
		y = self.conv1(x)
		y = self.bn1(y)
		y = self.relu(y)
		y = self.conv2(y)
		y = self.bn2(y)
		if self.is_downsample:
			x = self.downsample(x)
		return F.relu(x.add(y),True)

def make_layers(c_in,c_out,repeat_times, is_downsample=False):
	blocks = []
	for i in range(repeat_times):
		if i ==0:
			blocks += [BasicBlock(c_in,c_out, is_downsample=is_downsample),]
		else:
			blocks += [BasicBlock(c_out,c_out),]
	return nn.Sequential(*blocks)


class Net(nn.Module):
	def __init__(self, num_classes=751):
		super(Net,self).__init__()
		# 3 128 64
		self.conv = nn.Sequential(
			nn.Conv2d(3,64,3,stride=1,padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
			# nn.Conv2d(32,32,3,stride=1,padding=1),
			# nn.BatchNorm2d(32),
			# nn.ReLU(inplace=True),
			nn.MaxPool2d(3,2,padding=1),
		)
		# 32 64 32
		self.layer1 = make_layers(64,64,2,False)
		# 32 64 32
		self.layer2 = make_layers(64,128,2,True)
		# 64 32 16
		self.layer3 = make_layers(128,256,2,True)
		# 128 16 8
		self.layer4 = make_layers(256,512,2,True)
		# 256 8 4
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
		# 256 1 1 
		# self.reid = reid
		# self.classifier = nn.Sequential(
		# 	nn.Linear(512, 256),
		# 	nn.BatchNorm1d(256),
		# 	nn.ReLU(inplace=True),
		# 	nn.Dropout(),
		# 	nn.Linear(256, num_classes),
		# )
	
	def forward(self, x):
		x = self.conv(x)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = self.avgpool(x)
		x = x.view(x.size(0),-1)
		# B x 128
		# if self.reid:
		# 	x = x.div(x.norm(p=2,dim=1,keepdim=True))
		# 	return x
		# classifier
		# x = self.classifier(x)
		return x



# class Resnet18(nn.Module):
#     def __init__(self):
#         super(Resnet18, self).__init__()
#         # self.layer = resnet18(pretrained=False, num_classes=128)
#         resnet = resnet18(pretrained=True)
#         modules = list(resnet.children())[:-1]
#         # self.layer.fc = nn.Sequential(nn.Linear(512, 512))
#         self.backbone = nn.Sequential(*modules)
#         # self.layer.fc = nn.Sequential(
#         # nn.Linear(512, 512))

#         # for param in model.parameters():
#         #     param.requires_grad = False

#     def forward(self, x):
#         x = self.backbone(x)
#         x = x.view(x.shape[0], -1)
#         return x


class MobileNetv2(nn.Module):
	def __init__(self):
		super(MobileNetv2, self).__init__()
		model = mobilenet_v2(pretrained=True)
		self.layer1 = model.features

		# building classifier
		self.layer2 = nn.Sequential(
			nn.Linear(62720, 128),
		)

	def forward(self, x):
		x = self.layer1(x)
		x = x.view(x.size(0), -1)
		output = self.layer2(x)
		return output


class EmbeddingNet(nn.Module):
	def __init__(self):
		super(EmbeddingNet, self).__init__()
		self.convnet = nn.Sequential(nn.Conv2d(3, 32, 5), nn.PReLU(),
									 nn.MaxPool2d(2, stride=2),
									 nn.Conv2d(32, 64, 5), nn.PReLU(),
									 nn.MaxPool2d(2, stride=2))

		self.fc = nn.Sequential(nn.Linear(73984, 256),
								nn.PReLU(),
								nn.Linear(256, 256),
								nn.PReLU(),
								nn.Linear(256, 2)
								)

	def forward(self, x):
		output = self.convnet(x)
		output = output.view(output.size()[0], -1)
		output = self.fc(output)
		return output

	def get_embedding(self, x):
		return self.forward(x)


class EmbeddingNetL2(EmbeddingNet):
	def __init__(self):
		super(EmbeddingNetL2, self).__init__()

	def forward(self, x):
		output = super(EmbeddingNetL2, self).forward(x)
		output /= output.pow(2).sum(1, keepdim=True).sqrt()
		return output

	def get_embedding(self, x):
		return self.forward(x)


class ClassificationNet(nn.Module):
	def __init__(self, embedding_net, n_classes):
		super(ClassificationNet, self).__init__()
		self.embedding_net = embedding_net
		self.n_classes = n_classes
		self.nonlinear = nn.PReLU()
		self.fc1 = nn.Linear(2, n_classes)

	def forward(self, x):
		output = self.embedding_net(x)
		output = self.nonlinear(output)
		scores = F.log_softmax(self.fc1(output), dim=-1)
		return scores

	def get_embedding(self, x):
		return self.nonlinear(self.embedding_net(x))


class SiameseNet(nn.Module):
	def __init__(self, embedding_net):
		super(SiameseNet, self).__init__()
		self.embedding_net = embedding_net

	def forward(self, x1, x2):
		output1 = self.embedding_net(x1)
		output2 = self.embedding_net(x2)
		return output1, output2

	def get_embedding(self, x):
		return self.embedding_net(x)


class TripletNet(nn.Module):
	def __init__(self, embedding_net):
		super(TripletNet, self).__init__()
		self.embedding_net = embedding_net

	def forward(self, x1, x2, x3):
		output1 = self.embedding_net(x1)
		output2 = self.embedding_net(x2)
		output3 = self.embedding_net(x3)
		return output1, output2, output3

	def get_embedding(self, x):
		return self.embedding_net(x)


if (__name__ == "__main__"):
	net = Resnet18()
	x = torch.Tensor(1, 3, 64, 128)
	x = net(x)
	print(x.shape)
