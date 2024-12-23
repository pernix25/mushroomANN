import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# set up model
class ANN_model(nn.Module):

  # input layer -> 21
  # hidden layer 1 -> 32
  # hidden layer 2 -> 16
  # hidden layer 3 -> 8
  # output layer -> 2

  def __init__(self):
    super().__init__()
    self.hl1 = nn.Linear(in_features=21, out_features=32)
    self.hl2 = nn.Linear(in_features=32, out_features=16)
    self.hl3 = nn.Linear(in_features=16, out_features=8)
    self.out = nn.Linear(in_features=8, out_features=2)

  def forward(self, x):
    x = F.relu(self.hl1(x))
    x = F.relu(self.hl2(x))
    x = F.relu(self.hl3(x))
    x = self.out(x)

    return x

# set random seed and instantiate model
torch.manual_seed(42)
model = ANN_model()

# import dataset
df = pd.read_csv('mushroom_data')

# change label to 0,1
df['label'] = df['label'].replace('p', 1)
df['label'] = df['label'].replace('e', 0)

# seperate class label
X = df.drop('label', axis=1)
X = X.drop(' stalk root', axis=1)
y = df['label']

# get features from dataframe
X = X.values
y = y.values

# shuffle dataset
from sklearn.utils import shuffle
X, y = shuffle(X, y, random_state=42)

# convert characters into numbers
X = [[float(ord(char)) for char in lyst] for lyst in X]

# normalize data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# convert arrays to tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# set params
loss_func = nn.CrossEntropyLoss()
learn_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

# train model

epochs = 250
losses = []

for i in range(epochs):
  y_pred = model.forward(X_train)
  loss = loss_func(y_pred, y_train)
  losses.append(loss)

  if i%25 == 0:
    print(f'Epoch: {i} Loss: {loss}')

  loss.backward()
  optimizer.step()
  optimizer.zero_grad()

# graph
plt.plot(range(epochs), [loss.item() for loss in losses])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

# testing the model
preds = []
with torch.no_grad():
  for val in X_test:
    y_pred = model.forward(val)
    preds.append(y_pred.argmax().item())

# convert y_test to numpy for comparison
y_test = y_test.numpy()

# get accuracy from model test
accuracy = accuracy_score(y_test, preds)
print(f"Accuracy: {accuracy * 100:.2f}%")

# plot confusion matrix
cm = confusion_matrix(y_test, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()
