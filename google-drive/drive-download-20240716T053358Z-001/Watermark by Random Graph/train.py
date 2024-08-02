import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import NeighborSampler

from models import GraphSAGE, WatermarkGraph

'''
==Core Settings==
data split: [0.4, 0.2, 0.4] - TODO
aggregation: mean
sample: 5
layer: 2
hidden: 128
logit: softmax
loss: cross_entropy
optimizer: Adam
run 10 times average

==Questions==
train watermark neighbor sample?
'''

# Load the Cora dataset
dataset = Planetoid(root='~/Datasets', name='Cora')
data = dataset[0]

# Load Watermark Graph
wm_node = 50
pr = 0.1
pg = 0
data_wm = WatermarkGraph(n=wm_node, num_features=dataset.num_node_features, num_classes=dataset.num_classes, pr=pr, pg=pg).graph_wm

# Sample data
train_loader = NeighborSampler(data.edge_index, node_idx=data.train_mask,
                               sizes=[5, 5], batch_size=32, shuffle=True, num_nodes=data.num_nodes)
val_loader = NeighborSampler(data.edge_index, node_idx=data.val_mask,
                             sizes=[5, 5], batch_size=32, shuffle=False, num_nodes=data.num_nodes)
test_loader = NeighborSampler(data.edge_index, node_idx=data.test_mask,
                              sizes=[5, 5], batch_size=32, shuffle=False, num_nodes=data.num_nodes)
wm_loader = NeighborSampler(data_wm.edge_index, sizes=[5, 5], batch_size=wm_node, shuffle=False, num_nodes=wm_node)

# Load GraphSAGE model
model = GraphSAGE(in_channels=dataset.num_node_features, hidden_channels=128, out_channels=dataset.num_classes)

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


def train(loader):
    model.train()
    total_loss = 0
    for batch_size, n_id, adjs in loader:
        optimizer.zero_grad()
        out = model(data.x[n_id], adjs)
        loss = F.cross_entropy(out[:batch_size], data.y[n_id[:batch_size]])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def train_wm(loader):
    model.train()
    total_loss = 0
    for batch_size, n_id, adjs in loader:
        optimizer.zero_grad()
        out = model(data_wm.x[n_id], adjs)
        loss = F.cross_entropy(out[:batch_size], data_wm.y[n_id[:batch_size]])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def test(loader):
    model.eval()
    correct = 0
    for batch_size, n_id, adjs in loader:
        out = model(data.x[n_id], adjs)
        pred = out[:batch_size].max(1)[1]
        correct += pred.eq(data.y[n_id[:batch_size]]).sum().item()
    return correct / data.test_mask.sum().item()


def test_wm(loader):
    model.eval()
    correct = 0
    for batch_size, n_id, adjs in loader:
        out = model(data_wm.x[n_id], adjs)
        pred = out[:batch_size].max(1)[1]
        correct += pred.eq(data_wm.y[n_id[:batch_size]]).sum().item()
    return correct / len(data_wm.y)


for epoch in range(1, 51):
    loss = train(train_loader)
    val_acc = test(val_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')

nonmarked_acc = test(wm_loader)  # Table 2

for epoch in range(1, 11):
    loss = train_wm(wm_loader)
    test_acc = test_wm(wm_loader)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test Acc: {test_acc:.4f}')


# Final results
marked_acc = test(test_loader)  # Table 1
watermark_acc = test_wm(wm_loader)  # Table 3
print('Final results')
print(f'Non-Marked Acc: {nonmarked_acc:.4f}, Marked Acc: {marked_acc:.4f}, Watermark Acc: {watermark_acc:.4f}')


# TODO Model pruning
# TODO Model Fine-tuning
