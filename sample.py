import torch
 
x = torch.tensor([[1., 2., 3.]])
y = torch.tensor([[4., 5., 6.]])
# a = torch.linalg.norm(c, dim=1)

# x = x.div(x.norm(p=2,dim=1,keepdim=True))


# y = y.div(y.norm(p=2,dim=1,keepdim=True))

# print(x-y)


distance_positive = torch.norm((x-y), 2, dim=1)
print(distance_positive)