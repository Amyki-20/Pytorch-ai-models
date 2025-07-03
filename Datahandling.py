import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

digits = load_digits()
X, y = digits.data, digits.target
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 11)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
class CustomDataset(Dataset):
    def __init__(self, data, target):
        self.data = torch.tensor(data, dtype= torch.float32)
        self.target = torch.tensor(target, dtype=torch.long )
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        sample = {'data': self.data[index],'target': self.target[index]}
        return sample

train_dataset = CustomDataset(X_train, y_train)
test_dataset = CustomDataset(X_test, y_test)
train_data_loader = DataLoader(dataset= train_dataset, batch_size = 32, shuffle = True, num_workers=0)
test_data_loader = DataLoader(dataset = test_dataset, batch_size=32, shuffle=False, num_workers=0)

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x
input_size = X_train.shape[1]
hidden_size = 64
output_size = len(set(y_train))
model = SimpleNN(input_size, hidden_size , output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr= 0.01)
epochs = 100 #epoch = the model sees all training examples once
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for batch in train_data_loader:
        inputs = batch['data']
        targets = batch['target']
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch: {epoch+1}/{epochs}, Loss: {running_loss/len(train_data_loader)}")

model.eval()
all_prediction = []
all_targets = []

with torch.no_grad():
    for batch in test_data_loader:
        inputs = batch['data']
        targets = batch['target']
        outputs = model(inputs)
        predictions = torch.argmax(outputs,dim = 1)
        all_prediction.extend(predictions.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
    accuracy = accuracy_score(all_targets, all_prediction)
    print(f"Accuracy Test: {accuracy *100:.3f}%")















