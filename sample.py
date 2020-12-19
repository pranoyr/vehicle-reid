import torch
 
c = torch.tensor([[1., 2., 3.]])
a = torch.linalg.norm(c, dim=1)


distance_positive = torch.norm(c, 2, dim=1)
print(a)