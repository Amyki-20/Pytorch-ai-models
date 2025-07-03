import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader #Loads your dataset in mini-batches
from torchvision import datasets,transforms #transforms: Used to preprocess/convert images (e.g., to tensors)
import time
from torchvision.utils import make_grid
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
#convert MNIST image files to tensors of 4D(# of images, height, width, color channel)
transform = transforms.ToTensor()
#train data
train_data = datasets.MNIST(root = 'cnn_data', train = True, download = True, transform = transform)
test_data = datasets.MNIST(root = 'cnn_data', train = False, download = True, transform = transform)
print(test_data)
print(train_data)
#create a small batch-size for images........eg-10
train_loader = DataLoader(train_data, batch_size= 10, shuffle= True)
test_loader = DataLoader(test_data, batch_size = 10, shuffle = False)
#define CNN model
#describe convolutional layer and what it does (2 layers)
conv1 = nn.Conv2d(1, 6,3,1)
conv2 = nn.Conv2d(6,16,3,1)
#grab 1 mnist record/image
for i, (X_Train, y_train) in enumerate(train_data): #(X_Train, y_train): the image and its label
    break
print(X_Train.shape)
x = X_Train.view(1,1,28,28)
# perform first convolution
x = F.relu(conv1(x))
print(x)
print(x.shape)
# pass through pooling layer
x = F.max_pool2d(x,2,2)
print(x.shape)
# second convolution layer
x = F.relu(conv2(x))
print(x.shape)
#since we didnt set padding we lose pixels
#pooling layer
x = F.max_pool2d(x,2,2)
print(x.shape)
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,6,3,1)
        self.conv2 = nn.Conv2d(6,16,3,1)
        #fully connected layers
        self.fc1 = nn.Linear(5*5*16,120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear( 84, 10)
    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X,2,2)
        #second pass
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        #Re-view to flatten it out
        X = X.view(-1, 5*5*16)
        #fully connected layers
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X, dim = 1)
torch.manual_seed(41)
model = CNN()
print(model)
# loss function optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)
start_time = time.time()
epochs = 5
train_losses = []
test_losses = []
train_correct = []
test_correct = []
for epoch in range(epochs):
    trn_corr = 0
    tst_corr = 0
    for b, (X_train, y_train) in enumerate(train_loader):
        b += 1
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        predicted = torch.max(y_pred.data, 1)[1]
        batch_correct = (predicted == y_train).sum()
        trn_corr += batch_correct
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if b%600 == 0: #every 600 batch
            print(f"Epoch: {epoch}, Batch: {b}, Loss: {loss.item()}")
    train_losses.append(loss)
    train_correct.append(trn_corr)

    with torch.no_grad():
        for i, (X_test, y_test) in enumerate(test_loader): #enumerate adds an index counter (i)
            y_val = model(X_test)
            predicted = torch.max(y_val.data,1)[1]
            tst_corr += (predicted == y_test).sum()
            loss = criterion(y_val, y_test)
        test_losses.append(loss)
        test_correct.append(tst_corr)



current_time = time.time()
total_time = current_time - start_time
print(f"Training took: {total_time/60} minutes")

train_losses =[tl.item() for tl in train_losses]
test_losses =[tl.item() for tl in test_losses]
plt.plot(train_losses, label = 'Training Loss')
plt.plot(test_losses, label = 'Testing Loss')
plt.title('Loss at epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
test_load_everything = DataLoader(test_data, batch_size = 10000, shuffle = False)
with torch.no_grad():
    correct = 0
    for X_test, y_test in test_load_everything:
        y_eval = model(X_test)
        predicted = torch.max(y_eval, 1)[1]
        correct += (predicted == y_test).sum()
print(correct.item()/len(test_data) * 100)

#graph accuracy at the end of each epoch
plt.plot([t/len(train_data) for t in train_correct], label = 'training accuracy')
plt.plot([t/len(test_data) for t in test_correct], label = 'validation accuracy')
plt.title('Accuracy the end of each epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#grab an image
test_data[4143][0].reshape(28,28)
plt.imshow(test_data[4143][0].reshape(28,28))
model.eval()
with torch.no_grad():
    new_prediction = model(test_data[4143][0].view(1,1,28,28))
    print(new_prediction)
    print(new_prediction.argmax())














