import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import erdos_renyi_graph


# Define the GraphSAGE model
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr='mean')
        self.conv2 = SAGEConv(hidden_channels, out_channels, aggr='mean')

    def forward(self, x, adjs):
        x = self.conv1(x, adjs[0].edge_index)
        x = F.relu(x)
        x = self.conv2(x, adjs[1].edge_index)
        return F.softmax(x, dim=1)


class WatermarkGraph:
    def __init__(self, n, num_features, num_classes, pr=0.1, pg=0.1, device='cpu'):
        # X is 1 probability
        self.pr = pr
        # edge probability
        self.pg = pg
        # device
        self.device = device
        # generate
        self.graph_wm = self._generate_wm(n, num_features, num_classes)

    def _generate_wm(self, n, num_features, num_classes):
        # generate watermark graph with random edge
        wm_edge_index = erdos_renyi_graph(n, self.pg, directed=False)
        wm_edge_weight = torch.ones(wm_edge_index.shape[1], dtype=torch.float32)
        # generate node features
        wm_x = torch.tensor(np.random.binomial(1, self.pr, size=(n, num_features)),
                            dtype=torch.float32)
        # generate label
        wm_y = torch.tensor(np.random.randint(low=0, high=num_classes - 1, size=n), dtype=torch.long,
                            device=self.device)
        # Data
        data = Data(edge_index=wm_edge_index, x=wm_x, y=wm_y)
        return data
