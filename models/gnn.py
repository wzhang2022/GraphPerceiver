from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_geometric.utils import degree
from torch_scatter import scatter
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import to_dense_batch, to_dense_adj
import torch.nn.functional as F
from .helpers import Attention
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


def sparse_to_dense_batch(x, edge_index, edge_features, batch):
    node_features, node_mask = to_dense_batch(x, batch=batch)
    edge_batch = batch[edge_index[0]]

    edge_index_shifted, edge_mask = to_dense_batch(edge_index.transpose(0, 1), batch=edge_batch)
    edge_features, _ = to_dense_batch(edge_features, batch=edge_batch)
    batch_size, edge_batch_size = batch.max().item() + 1, edge_batch.max().item() + 1
    if edge_batch_size < batch_size:
        # Error handling if last graphs in batch have no edges
        edge_index_shifted= F.pad(edge_index_shifted, pad=(0, 0, 0, 0, 0, batch_size - edge_batch_size))
        edge_mask = F.pad(edge_mask, pad=(0, 0, 0, batch_size - edge_batch_size))
        edge_features = F.pad(edge_features, pad=(0, 0, 0, 0, 0, batch_size - edge_batch_size))

    one = batch.new_ones(batch.size(0))
    num_nodes = scatter(one, batch, dim=0, dim_size=batch_size, reduce='add')
    cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])[:-1]
    edge_index = edge_index_shifted - cum_nodes[..., None, None]
    edge_index = torch.masked_fill(edge_index, ~edge_mask.unsqueeze(-1), 0)
    return (node_features, edge_index, edge_features), (node_mask, edge_mask)


### GIN convolution along the graph structure
class GINConv(MessagePassing):
    def __init__(self, emb_dim):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GINConv, self).__init__(aggr = "add")

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

    def forward(self, x, edge_index, edge_embedding, batch):
        out = self.mlp((1 + self.eps) *x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))

        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


### GCN convolution along the graph structure
class GCNConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GCNConv, self).__init__(aggr='add')

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)

    def forward(self, x, edge_index, edge_embedding, batch):
        x = self.linear(x)

        row, col = edge_index

        #edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = degree(row, x.size(0), dtype = x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr = edge_embedding, norm=norm) + F.relu(x + self.root_emb.weight) * 1./deg.view(-1,1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class GNNAttentionBlock(torch.nn.Module):
    def __init__(self, emb_dim, num_heads, head_dim, gnn_type="gcn"):
        super(GNNAttentionBlock, self).__init__()
        if gnn_type=="gin":
            self.gnn = GINConv(emb_dim)
        elif gnn_type=="gcn":
            self.gnn = GCNConv(emb_dim)
        self.attn = Attention(query_dim=emb_dim, heads=num_heads, dim_head=head_dim)

    def forward(self, x, edge_index, edge_embedding, batch):
        x = self.gnn(x, edge_index, edge_embedding, batch)
        (node_features, edge_index, edge_features), (node_mask, edge_mask) = sparse_to_dense_batch(
            x, edge_index, edge_embedding, batch
        )
        attn_mask = torch.einsum("bi,bj->bij", node_mask, node_mask)
        x = self.attn(node_features, mask=attn_mask, mask_kv_only=False)
        x, edge_index, edge_attr, batch = dense_to_sparse(x, edge_index, edge_features, node_mask, edge_mask)
        return x



### GNN to generate node embedding
class GNN_node(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(
            self,
            num_layer,
            emb_dim,
            drop_ratio=0.5,
            JK="last",
            residual=False,
            gnn_type='gin',
            virtual_node=True):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers

        '''

        super(GNN_node, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        # add residual connection or not
        self.residual = residual
        self.virtual_node = virtual_node

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        if virtual_node:
            # set the initial virtual node embedding to 0.
            self.virtualnode_embedding = torch.nn.Embedding(1, emb_dim)
            torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

            # List of MLPs to transform virtual node at every layer
            self.mlp_virtualnode_list = torch.nn.ModuleList()
            for layer in range(num_layer - 1):
                self.mlp_virtualnode_list.append(
                    torch.nn.Sequential(
                        torch.nn.Linear(emb_dim, 2 * emb_dim),
                        torch.nn.BatchNorm1d(2 * emb_dim),
                        torch.nn.ReLU(),
                        torch.nn.Linear(2 * emb_dim, emb_dim),
                        torch.nn.BatchNorm1d(emb_dim),
                        torch.nn.ReLU()
                    )
                )

        ###List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layer):
            if gnn_type == 'gin':
                self.convs.append(GINConv(emb_dim))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(emb_dim))
            elif gnn_type == "gnn_attn":
                self.convs.append(GNNAttentionBlock(emb_dim, num_heads=4, head_dim=64, gnn_type="gcn"))
            else:
                raise ValueError('Undefined GNN type called {}'.format(gnn_type))

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, x, edge_index, edge_embedding, batch):
        if self.virtual_node:
            # virtual node embeddings for graphs
            virtualnode_embedding = self.virtualnode_embedding(
                torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))

        ### computing input node embedding

        h_list = [x]
        for layer in range(self.num_layer):
            if self.virtual_node:
                h_list[layer] = h_list[layer] + virtualnode_embedding[batch]

            h = self.convs[layer](h_list[layer], edge_index, edge_embedding, batch)
            h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)

            if self.residual:
                h += h_list[layer]

            h_list.append(h)

            if self.virtual_node and layer < self.num_layer - 1:
                ### add message from graph nodes to virtual nodes
                virtualnode_embedding_temp = global_add_pool(h_list[layer], batch) + virtualnode_embedding
                ### transform virtual nodes using MLP

                if self.residual:
                    virtualnode_embedding = virtualnode_embedding + F.dropout(self.mlp_virtualnode_list[layer](virtualnode_embedding_temp), self.drop_ratio, training = self.training)
                else:
                    virtualnode_embedding = F.dropout(self.mlp_virtualnode_list[layer](virtualnode_embedding_temp), self.drop_ratio, training = self.training)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]

        return node_representation

