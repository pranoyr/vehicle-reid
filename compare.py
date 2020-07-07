from losses import TripletLoss
from datasets import TripletMNIST, TripletVeriDataset
from networks import EmbeddingNet, Resnet18
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


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1):
        output1 = self.embedding_net(x1)
        return output1

    # def get_embedding(self, x):
    #     return self.embedding_net(x)


embedding_net=Resnet18()
model=TripletNet(embedding_net)
model.eval()
# model=embedding_net.to('cuda')


transform = transforms.Compose([
						   transforms.Resize((150,150)),
						   transforms.ToTensor(),
						   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
					        0.229, 0.224, 0.225])
					   ])


checkpoint = torch.load('/Users/pranoyr/Desktop/model6.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])

# img1 = cv2.imread('/Users/pranoyr/Downloads/VeRi/image_query/0219_c015_00027760_0.jpg')
# img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
# img1 = Image.fromarray(img1)

# img2 = cv2.imread('/Users/pranoyr/Downloads/VeRi/image_query/0219_c017_00027085_0.jpg')
# img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
# img2 = Image.fromarray(img2)

# img3 = cv2.imread('/Users/pranoyr/Downloads/VeRi/image_query/0218_c004_00043495_0.jpg')
# img3 = cv2.cvtColor(img3,cv2.COLOR_BGR2RGB)
# img3 = Image.fromarray(img3)



img1 = cv2.imread('/Users/pranoyr/Desktop/cars/entry.png')
img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
img1 = Image.fromarray(img1)

img2 = cv2.imread('/Users/pranoyr/Desktop/cars/exit.png')
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
img2 = Image.fromarray(img2)

img3 = cv2.imread('/Users/pranoyr/Desktop/cars/exit1.png')
img3 = cv2.cvtColor(img3,cv2.COLOR_BGR2RGB)
img3 = Image.fromarray(img3)


img1 = transform(img1)
img2 = transform(img2)
img3 = transform(img3)

with torch.no_grad():
    anchor = model(img1.unsqueeze(0))
    positive = model(img2.unsqueeze(0))
    negative = model(img3.unsqueeze(0))

print("positive")
distance_positive = (anchor - positive).pow(2).sum(1)
print(distance_positive)

print("negative")
distance_positive = (anchor - negative).pow(2).sum(1)
print(distance_positive)







