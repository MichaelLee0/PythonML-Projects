import torch

# Defining Tensors
# x = torch.ones(2,3, dtype = torch.double)
# x = torch.tensor([2.5, 1])

# Some basic operations
x = torch.tensor([1, 2, 3])
y = torch.tensor([1, 2, 3])

# These do the same thing  
# z = torch.add(x,y)
# z = x + y
y.add_(x)

print(x)
print(y)
# print(z)

