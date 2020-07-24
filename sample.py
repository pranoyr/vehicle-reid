from torchvision.models import mobilenet_v2
import torch.nn as nn

model = mobilenet_v2(pretrained = True)
new_classifier = nn.Sequential(*list(model.classifier.children())[:-1])
print(new_classifier)
