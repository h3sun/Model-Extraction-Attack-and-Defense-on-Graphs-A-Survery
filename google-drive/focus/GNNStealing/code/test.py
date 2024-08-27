import dgl
from dgl.data import CiteseerGraphDataset
from src.utils import load_data, split_graph_different_ratio
import networkx as nx
import numpy as np
from graphgallery.datasets import NPZDataset

data = NPZDataset('citeseer_full', verbose=False)
graph = data.graph
print(graph)

nx_g = nx.from_scipy_sparse_array(graph.adj_matrix)

for node_id, node_data in nx_g.nodes(data=True):
    node_data["features"] = graph.node_attr[node_id].astype(np.float32)
    node_data["labels"] = graph.node_label[node_id].astype(np.long)

dgl_graph = dgl.from_networkx(nx_g, node_attrs=['features', 'labels'])
dgl_graph = dgl.add_self_loop(dgl_graph)

print(dgl_graph)

g_list = dgl_graph.ndata['features'].numpy().tolist()

print(len(g_list))

train_subset, val_subset, test_subset = dgl.data.utils.split_dataset(
        g_list, frac_list=[0.6, 0.2, 0.2], shuffle=True)

print(train_subset, len(train_subset.indices))

train_index = train_subset.indices[:int(len(train_subset.indices) * 1.0)]
train_g = dgl_graph.subgraph(train_index)

print(train_g)
