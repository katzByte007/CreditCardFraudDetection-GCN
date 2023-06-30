import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load and preprocess the dataset
df = pd.read_csv('creditcard.csv')


print(df.columns)
# Drop the 'Class' column and use the remaining columns as features
X = df.drop('Class', axis=1).values
y = df['Class'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the features and labels to PyTorch tensors
x_train = torch.tensor(X_train, dtype=torch.float)
y_train = torch.tensor(y_train, dtype=torch.long)

# Construct the financial network graph (fully connected graph assumption)
num_nodes = x_train.shape[0]
edge_index = []
for i in range(num_nodes):
    for j in range(i+1, num_nodes):
        edge_index.append([i, j])
edge_index = torch.tensor(edge_index).t().contiguous()

# Create a Data object
data = Data(x=x_train, y=y_train, edge_index=edge_index)

# Define the Graph Convolutional Network (GCN) model
class GCNNet(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Train and evaluate the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GCNNet(num_features=X_train.shape[1], num_classes=len(np.unique(y_train)))
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.NLLLoss()

# Create DataLoader objects for training and validation
train_loader = DataLoader(data, batch_size=64, shuffle=True)
val_loader = DataLoader(data, batch_size=64, shuffle=False)

def train():
    model.train()
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        output = model(batch.x, batch.edge_index)
        loss = criterion(output, batch.y)
        loss.backward()
        optimizer.step()

def evaluate():
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            output = model(batch.x, batch.edge_index)
            pred = output.argmax(dim=1)
            acc = accuracy_score(batch.y.cpu().numpy(), pred.cpu().numpy())
            return acc

for epoch in range(200):
    train()

accuracy = evaluate()
print("Accuracy:", accuracy)
