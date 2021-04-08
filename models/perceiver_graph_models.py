import random
import torch
import torch.nn as nn
from models.perceiver import Perceiver
from models.padded_mol_encoder import PaddedAtomEncoder, PaddedBondEncoder
from einops import rearrange


def transform_graph_to_input(batch_X, device):
    node_features, edge_index, edge_features = batch_X
    return node_features.to(device)


class HIVModelNodeOnly(nn.Module):
    def __init__(self, atom_emb_dim, perceiver_depth):
        super(HIVModelNodeOnly, self).__init__()
        self.atom_encoder = PaddedAtomEncoder(emb_dim=atom_emb_dim)
        self.perceiver = Perceiver(
            input_dim=atom_emb_dim,
            depth=perceiver_depth,
            num_latents=128,
            latent_dim=256,
            cross_heads=2,
            latent_heads=8,
            cross_dim_head=8,
            latent_dim_head=8,
            num_classes=2,
            attn_dropout=0.,
            ff_dropout=0.,
            weight_tie_layers=False
        )

    def forward(self, batch_X, X_mask, device):
        node_features, edge_index, edge_features = batch_X
        x = self.atom_encoder(node_features.to(device))
        x = self.perceiver(x, mask=X_mask[0].to(device))
        return x


class HIVModel(nn.Module):
    def __init__(self, atom_emb_dim, bond_emb_dim, perceiver_depth):
        super(HIVModel, self).__init__()
        self.atom_encoder = PaddedAtomEncoder(emb_dim=atom_emb_dim)
        self.bond_encoder = PaddedBondEncoder(emb_dim=bond_emb_dim)
        self.perceiver = Perceiver(
            input_dim=2 * atom_emb_dim + bond_emb_dim,
            depth=perceiver_depth,
            num_latents=128,
            latent_dim=256,
            cross_heads=4,
            latent_heads=8,
            cross_dim_head=8,
            latent_dim_head=8,
            num_classes=2,
            attn_dropout=0,
            ff_dropout=0.2,
            weight_tie_layers=False
        )

    def forward(self, batch_X, X_mask, device):
        """
        :param batch_X: (bs, num_nodes, num_node_feat), (bs, num_edges, 2), (bs, num_edges, num_edge_feat)
        :param X_mask: (bs, num_nodes), (bs, num_edges)
        :param device: cuda or cpu
        """
        node_features, edge_index, edge_features = batch_X[0].to(device), batch_X[1].to(device), batch_X[2].to(device)
        bs, num_nodes, _ = node_features.shape
        num_edges = edge_features.shape[1]
        flat_node_features = rearrange(node_features, "b n f -> (b n) f")
        flat_edge_index = rearrange(edge_index, "b m e-> (b m) e")
        index_shift = torch.arange(bs).to(device).repeat_interleave(num_edges) * num_nodes
        flat_edge_index_adjusted_1 = flat_edge_index[:, 0] + index_shift
        flat_edge_index_adjusted_2 = flat_edge_index[:, 1] + index_shift
        flat_node_1 = torch.index_select(flat_node_features, dim=0, index=flat_edge_index_adjusted_1)
        flat_node_2 = torch.index_select(flat_node_features, dim=0, index=flat_edge_index_adjusted_2)
        node_1 = rearrange(flat_node_1, "(b m) f -> b m f", b=bs)
        node_2 = rearrange(flat_node_2, "(b m) f -> b m f", b=bs)
        x_1 = self.atom_encoder(node_1)
        x_2 = self.atom_encoder(node_2)
        x_3 = self.bond_encoder(edge_features)
        if random.randint(0, 1) == 0:
            x = torch.cat([x_1, x_2, x_3], dim=2)
        else:
            x = torch.cat([x_2, x_1, x_3], dim=2)
        x = self.perceiver(x, mask=X_mask[1].to(device))
        return x