from __future__ import print_function
import torch

x = torch.rand(5, 3, dtype=torch.float)
y = torch.zeros(5, 3)
z = x + y
h = torch.add(z, x)
print(type(h))
print(h)

print(torch.cuda.is_available())

