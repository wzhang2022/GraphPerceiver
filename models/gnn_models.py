import torch
import torch.nn as nn
from .gnn import GNN
from einops import rearrange


def dense_to_sparse(node_features, edge_index, edge_features, node_mask, edge_mask):
    bs, num_nodes, _ = node_features.shape
    x = node_features[node_mask]
    edge_attr = edge_features[edge_mask]
    batch = rearrange(torch.arange(bs).repeat_interleave(num_nodes), "(b n) -> b n", b=bs)[node_mask]
    batch = batch.to(x.device)
    num_nodes_per_batch = torch.sum(node_mask, dim=1)
    index_shift = torch.cumsum(num_nodes_per_batch, dim=0).roll(1)
    index_shift[0] = 0
    edge_index = (edge_index + index_shift[..., None, None])[edge_mask]
    edge_index = torch.transpose(edge_index, 0, 1)
    return x, edge_index, edge_attr, batch



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
        self.gnn = GNN(
            num_tasks,
            num_layer=num_layer,
            emb_dim=emb_dim,
            gnn_type=gnn_type,
            virtual_node=virtual_node,
            residual=residual,
            drop_ratio=drop_ratio,
            JK=JK,
            graph_pooling=graph_pooling
        )


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
        return self.gnn(x, edge_index, edge_attr, batch)
