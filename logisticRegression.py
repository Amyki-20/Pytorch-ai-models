import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from  sklearn.datasets import make_classification
from torch.nn import BCELoss

device = torch.device('cpu')
X, y = make_classification(
    n_samples= 1000,
    n_features= 17,
    n_informative= 10,
    n_redundant= 7,
    n_classes =  2,
    random_state= 21
)
#X is the features matrix:→ 1000 rows × 17 columns → (1000, 17)
#y is the labels vector:→ 1000 labels → (1000,)
print(X.shape)
n_samples, n_features = X.shape
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.2, random_state= 18)
sc = StandardScaler() # standardizes your features
#Each feature gets rescaled to have mean = 0 and standard deviation = 1
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# change numpy to tensor
X_train = torch.from_numpy(X_train).type(torch.float32).to(device)
X_test = torch.from_numpy(X_test).type(torch.float32).to(device)
y_train = torch.from_numpy(y_train).type(torch.float32).view(-1,1).to(device)
y_test = torch.from_numpy(y_test).type(torch.float32).view(-1,1).to(device)
class LogisticRegression(nn.Module):
    def __init__(self,n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear1 = nn.Linear(n_input_features, 20)
        self.linear2 = nn.Linear(20,1)
        self.elu = nn.ELU()
    def forward(self,x):
        x = self.elu(self.linear1(x)) # Apply ELU activation to hidden layer
        y_predicted = torch.sigmoid(self.linear2(x)) # Apply sigmoid to output
        #torch.sigmoid() = squashes output to (0, 1) so it can be interpreted as probability.
        return y_predicted

model = LogisticRegression(n_features).to(device)
learning_rate = 0.1
criterion = BCELoss() #binary
optimizer = torch.optim.SGD(model.parameters(),lr= learning_rate)
epochs = 1000
for epoch in range(epochs):
    model.train()
    y_predicted = model(X_train)
    loss = criterion(y_predicted, y_train)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if (epoch+1)%100 == 0:
        print(f"Epoch: {epoch+1} , loss: {loss.item():.2f}")

model.eval()
with torch.inference_mode():
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(y_test).sum() /float(y_test.shape[0])
    print(f"Accuracy: {acc:.2f}")




