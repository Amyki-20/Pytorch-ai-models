import torch
import numpy as np

tensor1 = torch.tensor([1,2,3])
print("Output", tensor1)
array1 = np.array([1,2,3])
tensor_from_numpy = torch.from_numpy(array1)
# to convert back tensor =  tensor_from_numpy.numpy()
# torch.from_numpy , numpy() will share memory, torch.tensor() will create new memory
print(tensor1[1: ])
print("Tensor from numpy: ", tensor_from_numpy)
empty_tensor = torch.empty(3)
print("Empty Tensor: ", empty_tensor)
empty_tensor.fill_(5)
print("Filled Empty Tensor: ",empty_tensor)
zeros = torch.zeros(2,3)
print("Zeros: ", zeros) #torch.ones()
rand = torch.rand(4,4)
print("Random: ", rand)
aranged_tensor = torch.arange(start= 0, end=20, step=10)
print("Aranged Tensor: ", aranged_tensor)
print(rand[  : ,0])
#array[:, n] extracts the n-th column for any 2D tensor.
#array[n, :] extracts the n-th row.
print(rand[1,1].item())
#items() is used to convert a single-element tensor into a Python scalar sincePyTorch tensors cannot be directly used in Python operations
reshaped_tensor = rand.view(-1, 2)
print("Reshaped Tensor: ", reshaped_tensor)
a = torch.tensor([[2,3],[4,5.5]], requires_grad=True)
b = torch.tensor([[5,6],[8.0,9]])
sum = torch.add(a,b) # b.add_(a) to modify original tensor b
print("Sum: ", sum)
sub = torch.sub(b,a)
print("Difference: ", sub)
div = torch.div(b,a)
print("Division: ", div)
mul = torch.mm(a,b) # matrix multiplication , torch.mul() elementwise multiplication mm- same as matmul for 2D only
print("Multiplication: ", mul)
print(a.sum())
print(rand.shape)
print("Before permuting: ",a)
print("After permuting: ",a.permute(1,0))