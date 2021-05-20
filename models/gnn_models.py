import torch
import torch.nn as nn
from .gnn import GNN_node, dense_to_sparse
from einops import rearrange
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder



class MoleculeGNNModel(nn.Module):
    def __init__(
            self,
            num_tasks,
            num_layer=5,
            emb_dim=300,
            gnn_type='gin',
            virtual_node=True,
            residual=False,
            drop_ratio=0.5,
            JK="last",
            graph_pooling="mean"
    ):
        super(MoleculeGNNModel, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim)
        self.bond_encoder = BondEncoder(emb_dim)

        self.gnn_node = GNN_node(
            num_layer,
            emb_dim,
            JK=JK,
            drop_ratio=drop_ratio,
            residual=residual,
            gnn_type=gnn_type,
            virtual_node=virtual_node
        )

        # Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps = 2)
        else:
            raise ValueError("Invalid graph pooling type.")

        if graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(2*self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_tasks)

    def forward(self, batch_X, X_mask, device):
        """
        :param batch_X: (bs, num_nodes, num_node_feat), (bs, num_nodes, num_node_preprocess_feat),
                        (bs, num_edges, 2), (bs, num_edges, num_edge_feat)
        :param X_mask: (bs, num_nodes), (bs, num_edges)
        :param device: cuda or cpu
        """
        node_features, node_preprocess_feat, edge_index, edge_features = [X.to(device) for X in batch_X]
        node_mask, edge_mask = [mask.to(device) for mask in X_mask]
        x,  edge_index, edge_attr, batch = dense_to_sparse(node_features, edge_index, edge_features, node_mask, edge_mask)
        x = self.atom_encoder(x)
        edge_embedding = self.bond_encoder(edge_attr)
        h_node = self.gnn_node(x, edge_index, edge_embedding, batch)
        h_graph = self.pool(h_node, batch)
        return self.graph_pred_linear(h_graph)