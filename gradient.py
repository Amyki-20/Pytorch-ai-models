import torch
x = torch.tensor([3.0], dtype =  torch.float32, requires_grad = True)
z = x**2 + 4
z.backward()
print("Gradient:x ", x.grad)
# x.requires.grad_(True) overwrites the one above
# or wrap in
# with no torch.grad():
#    stmnt
m = torch.tensor([4.0], dtype =  torch.float32, requires_grad = True)
b = torch.tensor([2.0], dtype =  torch.float32, requires_grad = True)
x =  x.detach()
y = m*x + b
y.backward()
print("Gradient:x ", x.grad)
print("Gradient:b  ", b.grad)
print("Gradient:m  ", m.grad)
