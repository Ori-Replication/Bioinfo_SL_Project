from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv, Linear,GINConv
import torch_geometric.transforms as T
import torch
import pandas as pd
import numpy as np
from  tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



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

# HGT Model with a classification head
class HGT4Classification(nn.Module):
    def __init__(self,args, hgt,emb_dim,hidden_dim, num_classes, len_unique_node):
        super(HGT4Classification, self).__init__()
        self.hgt = hgt
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
        self.args = args
    
    def forward(self, kg_batch,batch):
        node_rep = self.hgt(kg_batch.x_dict, kg_batch.edge_index_dict)
        node_rep=   node_rep[args.node_type]
        node_set=pd.DataFrame(list(kg_batch[node_type].n_id[:len_unique_node].squeeze().detach().cpu().numpy()))
        node_set.drop_duplicates(inplace=True,keep='first')
        node_set[1]=range(node_set.shape[0])
        node_map=dict(zip(node_set[0],node_set[1]))
        batch=pd.DataFrame(batch.numpy())
        prediction_edge=batch[[0,1]]
        prediction_label=batch[2]
        edge_a,edge_b=prediction_edge[0],prediction_edge[1]
        edge_a=edge_a.map(node_map)
        edge_b=edge_b.map(node_map)
        HGT_nodea_emb=node_rep[edge_a.values]
        HGT_nodeb_emb=node_rep[edge_b.values]
        edge_embedding = torch.cat([HGT_nodea_emb, HGT_nodeb_emb], dim=1)
        emb_dim = edge_embedding.size(1)
        pred = self.mlp(edge_embedding)
        return pred

class HGT_ESM_4Classification(nn.Module):
    def __init__(self,args, hgt,emb_dim,hidden_dim, num_classes, len_unique_node):
        super(HGT_ESM_4Classification, self).__init__()
        self.hgt = hgt
        self.mlp = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        self.args = args
        self.esm_linear_a = nn.Linear(1280, 256)
        self.esm_linear_b = nn.Linear(1280, 256)
    def forward(self, kg_batch,batch, ESM_nodea_emb, ESM_nodeb_emb):
        node_rep = self.hgt(kg_batch.x_dict, kg_batch.edge_index_dict)
        node_rep=   node_rep[args.node_type]
        node_set=pd.DataFrame(list(kg_batch[node_type].n_id[:len_unique_node].squeeze().detach().cpu().numpy()))
        node_set.drop_duplicates(inplace=True,keep='first')
        node_set[1]=range(node_set.shape[0])
        node_map=dict(zip(node_set[0],node_set[1]))
        batch=pd.DataFrame(batch.numpy())
        prediction_edge=batch[[0,1]]
        prediction_label=batch[2]
        edge_a,edge_b=prediction_edge[0],prediction_edge[1]
        edge_a=edge_a.map(node_map)
        edge_b=edge_b.map(node_map)
        HGT_nodea_emb=node_rep[edge_a.values]
        HGT_nodeb_emb=node_rep[edge_b.values]
        ESM_nodea_emb = self.esm_linear_a(ESM_nodea_emb)
        ESM_nodeb_emb = self.esm_linear_b(ESM_nodeb_emb)
        edge_embedding = torch.cat([HGT_nodea_emb, HGT_nodeb_emb, ESM_nodea_emb, ESM_nodeb_emb], dim=1)        
        emb_dim = edge_embedding.size(1)
        pred = self.mlp(edge_embedding) 
        return pred
    
    
class HGT_ESM_Attention_4Classification(nn.Module):
    def __init__(self, args, hgt):
        super(HGT_ESM_Attention_4Classification, self).__init__()
        self.hgt = hgt
        self.args = args
        self.esm_linear_a = nn.Linear(1280, 256)
        self.esm_linear_b = nn.Linear(1280, 256)
        
        # Attention Layer
        self.attention = nn.MultiheadAttention(embed_dim=512 + 256, num_heads=8)
        
        # Final Linear Layer
        self.fc = nn.Linear(512 + 256, 2)
        
    def forward(self, kg_batch, batch, ESM_nodea_emb, ESM_nodeb_emb):
        node_rep = self.hgt(kg_batch.x_dict, kg_batch.edge_index_dict)
        node_rep = node_rep[self.args.node_type]
        
        node_set = pd.DataFrame(list(kg_batch[self.args.node_type].n_id[:len_unique_node].squeeze().detach().cpu().numpy()))
        node_set.drop_duplicates(inplace=True, keep='first')
        node_set[1] = range(node_set.shape[0])
        node_map = dict(zip(node_set[0], node_set[1]))
        
        batch = pd.DataFrame(batch.numpy())
        prediction_edge = batch[[0, 1]]
        prediction_label = batch[2]
        
        edge_a, edge_b = prediction_edge[0], prediction_edge[1]
        edge_a = edge_a.map(node_map)
        edge_b = edge_b.map(node_map)
        
        HGT_nodea_emb = node_rep[edge_a.values]
        HGT_nodeb_emb = node_rep[edge_b.values]
        
        ESM_nodea_emb = self.esm_linear_a(ESM_nodea_emb)
        ESM_nodeb_emb = self.esm_linear_b(ESM_nodeb_emb)
        
        edge_embedding = torch.cat([HGT_nodea_emb, HGT_nodeb_emb, ESM_nodea_emb, ESM_nodeb_emb], dim=1)
        
        # Reshape for MultiheadAttention (batch_size, seq_length, embedding_dim)
        edge_embedding = edge_embedding.unsqueeze(0)  # Add batch dimension
        attn_output, _ = self.attention(edge_embedding, edge_embedding, edge_embedding)
        attn_output = attn_output.squeeze(0)  # Remove batch dimension
        
        pred = self.fc(attn_output)
        return pred
    

class HGT_ESM_CLdata_4Classification(nn.Module):
    def __init__(self,args,hgt):
        super(HGT_ESM_CLdata_4Classification, self).__init__()
        self.hgt = hgt
        self.esm_linear_a = nn.Linear(1280, 256)
        self.esm_linear_b = nn.Linear(1280, 256)
        self.fc1 = nn.Linear(768, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128 + 6, 2)
        self.args = args

    def forward(self,node_type, len_unique_node, kg_batch,batch, ESM_nodea_emb, ESM_nodeb_emb,cell_line_gene_data_nodea, cell_line_gene_data_nodeb):
        node_rep = self.hgt(kg_batch.x_dict, kg_batch.edge_index_dict)
        node_rep=   node_rep[node_type]
        node_set=pd.DataFrame(list(kg_batch[node_type].n_id[:len_unique_node].squeeze().detach().cpu().numpy()))
        node_set.drop_duplicates(inplace=True,keep='first')
        node_set[1]=range(node_set.shape[0])
        node_map=dict(zip(node_set[0],node_set[1]))
        batch=pd.DataFrame(batch.numpy())
        prediction_edge=batch[[0,1]]
        prediction_label=batch[2]
        edge_a,edge_b=prediction_edge[0],prediction_edge[1]
        edge_a=edge_a.map(node_map)
        edge_b=edge_b.map(node_map)
        HGT_nodea_emb=node_rep[edge_a.values]
        HGT_nodeb_emb=node_rep[edge_b.values]
        ESM_nodea_emb = self.esm_linear_a(ESM_nodea_emb)
        ESM_nodeb_emb = self.esm_linear_b(ESM_nodeb_emb)
        edge_embedding = torch.cat([HGT_nodea_emb, HGT_nodeb_emb, ESM_nodea_emb, ESM_nodeb_emb], dim=1)   
        edge_embedding = self.fc1(edge_embedding)
        edge_embedding = self.relu(edge_embedding)
        edge_embedding = torch.cat([edge_embedding, cell_line_gene_data_nodea, cell_line_gene_data_nodeb], dim=1)
        pred = self.fc2(edge_embedding)
        return pred
