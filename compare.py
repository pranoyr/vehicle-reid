from losses import TripletLoss
from datasets import TripletMNIST, TripletVeriDataset
from networks import EmbeddingNet, Net, MobileNetv2
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


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1):
        output1 = self.embedding_net(x1)
        return output1

        
def _pdist(a, b):
    """Compute pair-wise squared distance between points in `a` and `b`.

    Parameters
    ----------
    a : array_like
        An NxM matrix of N samples of dimensionality M.
    b : array_like
        An LxM matrix of L samples of dimensionality M.

    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that eleement (i, j)
        contains the squared distance between `a[i]` and `b[j]`.

    """
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a2, b2 = np.square(a).sum(axis=1), np.square(b).sum(axis=1)
    r2 = -2. * np.dot(a, b.T) + a2[:, None] + b2[None, :]
    r2 = np.clip(r2, 0., float(np.inf))
    return r2



def _nn_euclidean_distance(x, y):
    """ Helper function for nearest neighbor distance metric (Euclidean).

    Parameters
    ----------
    x : ndarray
        A matrix of N row-vectors (sample points).
    y : ndarray
        A matrix of M row-vectors (query points).

    Returns
    -------
    ndarray
        A vector of length M that contains for each entry in `y` the
        smallest Euclidean distance to a sample in `x`.

    """
    distances = _pdist(x, y)
    return np.maximum(0.0, distances.min(axis=0))



opt = parse_opts()
device = torch.device(f"cuda:{opt.gpu}" if opt.use_cuda else "cpu")

embedding_net=Net()
# embedding_net=MobileNetv2()
model=TripletNet(embedding_net)
model=model.to(device)
model.eval()

transform = transforms.Compose([
						   transforms.Resize((96,96)),
						   transforms.ToTensor(),
						   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
					        0.229, 0.224, 0.225])
					   ])

checkpoint = torch.load('/Users/pranoyr/Desktop/weights/car_re_id_model.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])


img1 = cv2.imread('/Users/pranoyr/Desktop/reid/6.png')
img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
img1 = Image.fromarray(img1)

img2 = cv2.imread('/Users/pranoyr/Desktop/reid/7.png')
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
img2 = Image.fromarray(img2)

img3 = cv2.imread('/Users/pranoyr/Desktop/reid/r2.png')
img3 = cv2.cvtColor(img3,cv2.COLOR_BGR2RGB)
img3 = Image.fromarray(img3)

img1 = transform(img1)
img2 = transform(img2)
img3 = transform(img3)

with torch.no_grad():
    anchor = model(img1.unsqueeze(0))
    positive = model(img2.unsqueeze(0))
    negative = model(img3.unsqueeze(0))

distance_positive = torch.norm(anchor - positive, 2, dim=1)
print(f'positive distance : {distance_positive}')

distance_negative = torch.norm(anchor - negative, 2, dim=1)
#distance_negative = (anchor - negative).pow(2).sum(1)
print(f'negative distance : {distance_negative}')

# positive = positive.div(positive.norm(p=2,dim=1,keepdim=True))
# anchor = anchor.div(anchor.norm(p=2,dim=1,keepdim=True))
# negative = negative.div(negative.norm(p=2,dim=1,keepdim=True))

# distances = _pdist(anchor, positive)
# distance_positive = np.maximum(0.0, distances.min(axis=0))
# print(f'positive distance : {distance_positive}')


# distances = _pdist(anchor, negative)
# distance_negative = np.maximum(0.0, distances.min(axis=0))
# # distance_negative = torch.norm(anchor - negative, 2, dim=1)
# #distance_negative = (anchor - negative).pow(2).sum(1)
# print(f'negative distance : {distance_negative}')



# positive = positive.div(positive.norm(p=2,dim=1,keepdim=True))
# anchor = anchor.div(anchor.norm(p=2,dim=1,keepdim=True))
# negative = negative.div(negative.norm(p=2,dim=1,keepdim=True))



# cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
# distance_positive = cosine_similarity(anchor, positive)
# print(f'positive distance : {distance_positive}')


# distance_negative = cosine_similarity(anchor, negative)
# # distance_negative = np.maximum(0.0, distances.min(axis=0))
# print(f'negative distance : {distance_negative}')