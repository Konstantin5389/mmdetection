import torch
import numpy as py

a = torch.rand(2, 2, 3)
print(a)
b = a.numpy()
print(b)
#print(b.shape)
# c = torch.from_numpy(b)
# print(c)
# print(torch.equal(a, c))
print(a[1, 0, 2])
print(b[1, 0, 2])