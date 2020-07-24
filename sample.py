from torchvision.models import mobilenet_v2
import torch.nn as nn

model = mobilenet_v2(pretrained = True)
print(model)

for params in model.parameters():
    params.requires_grad = False

