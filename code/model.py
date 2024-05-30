from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv, Linear,GINConv
import torch_geometric.transforms as T
import torch
import pandas as pd
import numpy as np
from  tqdm import tqdm
import torch.nn.functional as F




class HGT(torch.nn.Module):
    """The Heterogeneous Graph Sampler from the `"Heterogeneous Graph
    Transformer" <https://arxiv.org/abs/2003.01332>`_ paper.
    This loader allows for mini-batch training of GNNs on large-scale graphs
    where full-batch training is not feasible.

    Args:
        data (Any): A :class:`~torch_geometric.data.Data`,
            :class:`~torch_geometric.data.HeteroData`, or
            (:class:`~torch_geometric.data.FeatureStore`,
            :class:`~torch_geometric.data.GraphStore`) data object.
        num_samples (List[int] or Dict[str, List[int]]): The number of nodes to
            sample in each iteration and for each node type.
            If given as a list, will sample the same amount of nodes for each
            node type.
        input_nodes (str or Tuple[str, torch.Tensor]): The indices of nodes for
            which neighbors are sampled to create mini-batches.
            Needs to be passed as a tuple that holds the node type and
            corresponding node indices.
            Node indices need to be either given as a :obj:`torch.LongTensor`
            or :obj:`torch.BoolTensor`.
            If node indices are set to :obj:`None`, all nodes of this specific
            type will be considered.

    """
    def __init__(self, data,hidden_channels, out_channels, num_heads, num_layers):
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, data.metadata(),
                           num_heads)
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x.float()).relu_()

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
        # output node representation 
        for node_type in x_dict.keys():
            x_dict[node_type]=self.lin(x_dict[node_type])
        return x_dict


