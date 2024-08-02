from models import WatermarkGraph
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import NeighborSampler


dataset = Planetoid(root='~/Datasets', name='Cora')
data = dataset[0]
data_wm = WatermarkGraph(n=50, num_features=dataset.num_node_features, num_classes=dataset.num_classes).graph_wm
wm_node = 50
wm_loader = NeighborSampler(data_wm.edge_index, sizes=[5, 5], batch_size=wm_node, shuffle=False, num_nodes=wm_node)

print(data_wm)
print(wm_loader.sizes, len(wm_loader))
