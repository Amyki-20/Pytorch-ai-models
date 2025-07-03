import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

X = np.array([x for x in range(0,100)])
X = X.reshape(-1,1)
y = 46 + 2* X.flatten()
plt.scatter(X,y, label = "Initial Data")
plt.title("Pre Pytorch")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()
x_mean = X.mean()
x_std = X.std()
X_normalized =(X-x_mean)/x_std
X_tensor = torch.tensor(X_normalized, dtype= torch.float32)
print(X_tensor.shape)
y_mean= y.mean()
y_std = y.std()
y_normalized = (y - y_mean)/y_std
y_tensor = torch.tensor(y_normalized, dtype = torch.float32)
print(y_tensor.shape)

class LinearRegressionModel(nn.Module):
    def __init__(self, in_features, out_features):
         super().__init__()
         self.linear = nn.Linear(in_features, out_features)
    def forward(self, x):
        return self.linear(x).squeeze(1)
in_features = 1
out_features = 1
model = LinearRegressionModel(in_features, out_features)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
num_epochs = 100
for epoch in range(num_epochs):
    # forward pass
    outputs = model(X_tensor)
    # calculator loss
    loss = criterion(outputs, y_tensor)
    # backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

new_x = 121
new_x_normalized = (new_x-x_mean)/x_std
new_x_tensor = torch.tensor(new_x_normalized, dtype = torch.float32).view(1,-1)
model.eval()
with torch.no_grad():
    prediction_normalized = model(new_x_tensor)

prediction_denormalized = prediction_normalized.item() * y_std + y_mean
print(prediction_denormalized)
plt.scatter(X,y, label = "Initial Data")
fit_line = model(X_tensor).detach().numpy()* y_std + y_mean
plt.plot(X, fit_line,"r", label= "Pytorch Line")
plt.legend()
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Pytorch with Predictions")
plt.show()
