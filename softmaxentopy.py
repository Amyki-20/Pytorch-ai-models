import torch
import torch.nn as nn
from torch.onnx.symbolic_opset12 import cross_entropy_loss

logits = torch.tensor([[2.5, 0.2,0.6 ],
                       [0.3, 2.7,0.3 ],
                       [0.1, 0.2,1.6 ]])
target = torch.tensor([0,1,2])
softmax_prob = torch.softmax(logits, dim=1)
print(softmax_prob)
# manual_loss = torch.mean(-torch.log(softmax_prob.gather(1, target.view(-1,1)))) but their is an easier way
criterion = nn.CrossEntropyLoss()
loss = criterion(logits, target)
print(loss.item())

