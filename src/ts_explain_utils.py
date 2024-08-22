import torch.nn as nn
import torch


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)

X = torch.FloatTensor([1,1,1])
y = X.get_device()
print(y)
print(type(y))