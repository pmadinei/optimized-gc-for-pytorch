# Libraries
import torch
import torch.nn as nn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import torch.nn.functional as f
from sklearn.metrics import accuracy_score, confusion_matrix , precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

# Load Iris dataset from Sklearn
iris = load_iris()
x = iris.data
y = iris.target


# Split dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

x_train=torch.FloatTensor(x_train)
x_test=torch.FloatTensor(x_test)
y_train=torch.LongTensor(y_train)
y_test=torch.LongTensor(y_test)


# Model Neural Network
class model (nn.Module):
    def __init__(self,in_feat=4,h1=12,h2=12,h3=7,out_feat=3):
        super().__init__()
        self.layr1=nn.Linear(in_feat,h1)
        self.layr2=nn.Linear(h1,h2)
        self.layr3=nn.Linear(h2,h3)
        self.outlayr=nn.Linear(h3,out_feat)
    def forward(self,inpts):
        out1=f.relu(self.layr1(inpts))
        out2=f.relu(self.layr2(out1))
        out3=f.relu(self.layr3(out2))
        return self.outlayr(out3)

# Store Weights
losses = []
accuracies = []

# Optimizer
net = model()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.005) # lr = learning rate

# Training
for epoch in range(10000):
    optimizer.zero_grad()
    outputs = net(x_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 5000 == 0:
        print(f'Epoch [{epoch + 1}], Loss: {loss.item():.4f}')

    with torch.no_grad():
        test_outputs = net(x_test)
        _, predicted = torch.max(test_outputs, 1)
        accuracy = accuracy_score(y_test, predicted)
        accuracies.append(accuracy)

    losses.append(loss.item())

# Testing
with torch.no_grad():
    test_outputs = net(x_test)
    _, predicted = torch.max(test_outputs, 1)
    
    # precision / recall / f1 score
    precision = precision_score(y_test, predicted, average='weighted')
    recall = recall_score(y_test, predicted, average='weighted')
    f1 = f1_score(y_test, predicted, average='weighted')

print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Plot epoch loss and accuracy graph
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(len(losses)), losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')

plt.subplot(1, 2, 2)
plt.plot(range(len(accuracies)), accuracies, label='Test Accuracy', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Test Accuracy Over Time')

plt.show()

import objgraph

objgraph.show_refs(net, refcounts=True)
# objgraph.show_backrefs([net], refcounts=True)
# objgraph.show_backrefs(net, refcounts=True)

# roots = objgraph.get_leaking_objects()

# objgraph.show_most_common_types(objects=roots)