import torch
import numpy as np
from losses import TripletLoss
from datasets import TripletMNIST, TripletVeriDataset
from networks import EmbeddingNet
from metrics import AccumulatedAccuracyMetric
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
import torch.nn as nn
from PIL import Image
import torch.optim as optim
import os


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1):
        output1 = self.embedding_net(x1)
        return output1

    # def get_embedding(self, x):
    #     return self.embedding_net(x)



model=TripletNet(EmbeddingNet())
# model=embedding_net.to('cuda')


transform = transforms.Compose([
						   transforms.Resize((150,150)),
						   transforms.ToTensor(),
						   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
					        0.229, 0.224, 0.225])
					   ])


checkpoint = torch.load('/Users/pranoyr/Desktop/model8.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])

img1 = Image.open('/Users/pranoyr/Downloads/VeRi/image_query/0249_c017_00020505_0.jpg')
img2 = Image.open('/Users/pranoyr/Downloads/VeRi/image_query/0249_c019_00020775_0.jpg')
img3 = Image.open('/Users/pranoyr/Downloads/VeRi/image_query/0247_c016_00082105_0.jpg')

img1 = transform(img1)
img2 = transform(img2)
img3 = transform(img3)

anchor = model(img1.unsqueeze(0))
positive = model(img2.unsqueeze(0))
negative = model(img3.unsqueeze(0))

print("positive")
distance_positive = (anchor - positive).pow(2).sum(1)
print(distance_positive)

print("negative")
distance_positive = (anchor - negative).pow(2).sum(1)
print(distance_positive)







