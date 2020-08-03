from losses import TripletLoss
from datasets import TripletMNIST, TripletVeriDataset
from networks import EmbeddingNet, Resnet18, MobileNetv2
from metrics import AccumulatedAccuracyMetric
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

import cv2
import torch.nn as nn
from PIL import Image
import torch.optim as optim
import os
import torch
import numpy as np
from opts import parse_opts


# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1):
        output1 = self.embedding_net(x1)
        return output1


opt = parse_opts()
device = torch.device(f"cuda:{opt.gpu}" if opt.use_cuda else "cpu")

embedding_net=Resnet18()
# embedding_net=MobileNetv2()
model=TripletNet(embedding_net)
model=model.to(device)
model.eval()

transform = transforms.Compose([
						   transforms.Resize((224,224)),
						   transforms.ToTensor(),
						   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
					        0.229, 0.224, 0.225])
					   ])

checkpoint = torch.load('./snapshots/model2.pth')
model.load_state_dict(checkpoint['model_state_dict'])


img1 = cv2.imread('./images/swi.png')
img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
img1 = Image.fromarray(img1)

img2 = cv2.imread('./images/wag.png')
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
img2 = Image.fromarray(img2)

img3 = cv2.imread('./images/red1.png')
img3 = cv2.cvtColor(img3,cv2.COLOR_BGR2RGB)
img3 = Image.fromarray(img3)

img1 = transform(img1)
img2 = transform(img2)
img3 = transform(img3)

img1 = img1.unsqueeze(0)
img2 = img2.unsqueeze(0)
img3 = img3.unsqueeze(0)

img1 = img1.to(device)
img2 = img2.to(device)
img3 = img3.to(device)

with torch.no_grad():
    anchor = model(img1)
    positive = model(img2)
    negative = model(img3)

distance_positive = torch.norm(anchor - positive, 2, dim=1)
print(f'positive distance : {distance_positive}')

distance_negative = torch.norm(anchor - negative, 2, dim=1)
#distance_negative = (anchor - negative).pow(2).sum(1)
print(f'negative distance : {distance_negative}')