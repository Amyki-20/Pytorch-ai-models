import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# transform applies preprocessing like normalization and conversion to tensors.
class TabularDataset(Dataset):
    def __init__(self, data, transform = None):
        self.data = data
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self,index):
        sample = self.data[index]
        if self.transform:
            sample = self.transform(sample)
        return sample
class ToTensor:
    def __call__(self,sample):
        features, label = sample[0], sample[1]
        #Features = Input Data
        #These are the characteristics, measurements, or attributes you feed into your model to make predictions.
        # Labels = Output / Target
        return {'features' : torch.tensor(features,dtype = torch.float32),
                'label': torch.tensor(label, dtype=torch.float32)}
class Normalize:
    def __call__(self, sample):
        features, label = sample[0], sample[1]
        normalised_features = (features - np.mean(features))/np.std(features)
        return (normalised_features, label)

tabular_data = [(np.random.rand(2), np.random.rand()) for _ in range(100)]
transform = transforms.Compose([Normalize(), ToTensor()])
dataset = TabularDataset( tabular_data, transform)
dataloader = DataLoader(dataset,batch_size = 16, shuffle = True)

class SimpleModel(nn.Module):
    def __init__(self, input_size):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(input_size,1)

    def forward(self,x):
        return self.fc(x)

model = SimpleModel(2)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01)
epochs = 50
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        features, labels = batch['features'], batch['label']
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels.view(-1,1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    average_loss = total_loss/ len(dataloader)
    print(f"Epoch: {epoch+1}/{epochs} loss: {average_loss}")

model.eval()
with torch.no_grad():
    total_loss = 0
    for batch in dataloader:
        features, labels = batch['features'], batch['label']
        outputs = model(features)
        loss = criterion(outputs, labels.view(-1,1))
        total_loss += loss.item()
    average_loss = total_loss/len(dataloader)
    print(average_loss)


















