#Iris Project
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


class Model(nn.Module):
    def __init__(self, in_features=4,h1=8, h2=9, out_features=3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)
    def forward(self, x):
        # rectified linear Unit
        x = F.relu(self.fc1(x)) # do something then if output<0 use 0 else use output
        x = F.relu(self.fc2(x))
        x = F.relu(self.out(x))
        return x
#select a seed for randomization
torch.manual_seed(41)
#create an instance of our model
model = Model()
url = "https://gist.githubusercontent.com/Thanatoz-1/9e7fdfb8189f0cdf5d73a494e4a6392a/raw/aaecbd14aeaa468cd749528f291aa8a30c2ea09e/iris_dataset.csv"
my_df = pd.read_csv(url)
# my_df- to see all, my_df.head() - to see first few things u can add parameter
#my_df.tail() -to see final few things
my_df['target'] = my_df['target'].replace({
    'Iris-setosa': 0,
    'Iris-versicolor': 1,
    'Iris-virginica': 2
}).astype(int)
X = my_df.drop('target', axis=1)
y = my_df['target'] # All columns except 'target'
#convert to numpy values
X = X.values
y = y.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.tensor(y_train)
y_test = torch.tensor(y_test)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01) #lower the learning rate, training takes longer
epochs = 100
losses = []
for epoch in range(epochs):
    #go forward and get a prediction
    y_pred = model.forward(X_train)
    #measure loss
    loss = criterion(y_pred, y_train) # predicted value vs trained value
    losses.append(loss.detach())
    #print every 10 epochs
    if epoch % 10 == 0:
        print(f"Epoch: {epoch}, loss: {loss}")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
plt.plot(range(epochs), losses)
plt.title("IRIS")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend()
plt.show()
with torch.no_grad():
    y_eval = model.forward(X_test)
    loss = criterion(y_eval, y_test)
    print(loss)
correct = 0
with torch.no_grad():
    for i, data in enumerate(X_test):
        y_val = model.forward(data)
        print(f"{i+1}. {y_val} \t {y_test[i]} \t {y_val.argmax().item()}")
        # correct or not
        if y_val.argmax().item() == y_test[i]:
            print(f"Correct: {i+1}")
            correct += 1
print(f"We have {correct} amounts that are correct")
#During training: The model sees X_train and learns to match it to y_train (correct answers).
#During testing: You feed in X_test (which it has never seen) and check how close the predictions are to y_test (correct answers).
new_iris = torch.tensor([4.7, 3.2,1.3,0.2])
newer_iris = torch.tensor([4.6,3.6,1.0,0.2])
with torch.no_grad():
    #same as below (model(new_iris))
    y_new = model.forward(new_iris)
    print(y_new)
    print(model(newer_iris))
#torch.save(model.state_dict(), 'model.pth')  # `.pth` or `.pt` is common
#saving your trained model in PyTorch — which is essential when you're done training and want to reuse the model later without retraining.
#model.state_dict() gets a dictionary containing only the model’s parameters (weights and biases).
### TO RELOAD
#new_model = Model()  # You must recreate the same architecture
#new_model.load_state_dict(torch.load('model.pth'))
#new_model.eval()  # Set to evaluation mode
##CAN BE USED IN ANOTHER FILE IF
#1)You import PyTorch (import torch)
#2)You define the same model architecture (i.e., the same Model class)
#3)You load the weights from the saved file


















