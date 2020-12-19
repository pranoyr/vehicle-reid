import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from torchvision.models import mobilenet_v2
import torch


class Resnet18(nn.Module):
    def __init__(self):
        super(Resnet18, self).__init__()
        # self.layer = resnet18(pretrained=False, num_classes=128)
        resnet = resnet18(pretrained=True)
        modules = list(resnet.children())[:-1]
        # self.layer.fc = nn.Sequential(nn.Linear(512, 512))
        self.backbone = nn.Sequential(*modules)
        # self.layer.fc = nn.Sequential(
        # nn.Linear(512, 512))

        # for param in model.parameters():
        #     param.requires_grad = False

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.shape[0], -1)
        return x


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
