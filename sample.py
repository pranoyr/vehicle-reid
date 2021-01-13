import torch
import torch.nn as nn

input1 = torch.randn(100, 128)
input2 = torch.randn(100, 128)
cos = nn.CosineSimilarity(dim=1, eps=1e-6)
output = cos(input1, input2)

print(output.shape)



# distance_positive = (input1 - input2).pow(2).sum(1)  # .pow(.5)
# print(distance_positive.shape)


# losses = F.relu(distance_positive - distance_negative + self.margin)