import random
import torch
import torch.nn as nn
from models.perceiver import Perceiver, TransformerEncoder
from models.padded_mol_encoder import PaddedAtomEncoder, PaddedBondEncoder
from einops import rearrange


def get_node_feature_pairs(edge_index, node_encodings, device):
    bs, num_nodes, _ = node_encodings.shape
    num_edges = edge_index.shape[1]
    flat_node_encodings = rearrange(node_encodings, "b n f -> (b n) f")
    flat_edge_index = rearrange(edge_index, "b m e-> (b m) e")
    index_shift = torch.arange(bs).to(device).repeat_interleave(num_edges) * num_nodes
    flat_edge_index_adjusted_1 = flat_edge_index[:, 0] + index_shift
    flat_edge_index_adjusted_2 = flat_edge_index[:, 1] + index_shift
    flat_x_1 = torch.index_select(flat_node_encodings, dim=0, index=flat_edge_index_adjusted_1)
    flat_x_2 = torch.index_select(flat_node_encodings, dim=0, index=flat_edge_index_adjusted_2)

    x_1 = rearrange(flat_x_1, "(b m) f -> b m f", b=bs)
    x_2 = rearrange(flat_x_2, "(b m) f -> b m f", b=bs)
    return x_1, x_2

class HIVModelNodeOnly(nn.Module):
    def __init__(self, atom_emb_dim, perceiver_depth):
        super(HIVModelNodeOnly, self).__init__()
        self.atom_encoder = PaddedAtomEncoder(emb_dim=atom_emb_dim)
        self.perceiver = Perceiver(
            input_dim=atom_emb_dim,
            depth=perceiver_depth,
            latent_trnsfmr_depth=2,
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


class HIVPerceiverModel(nn.Module):
    def __init__(self, atom_emb_dim, bond_emb_dim,  node_preprocess_dim,
                 p_depth, p_latent_trsnfmr_depth, p_num_latents, p_latent_dim, p_cross_heads, p_latent_heads,
                 p_cross_dim_head, p_latent_dim_head, p_attn_dropout, p_ff_dropout, p_weight_tie_layers):
        super(HIVPerceiverModel, self).__init__()
        self.atom_encoder = PaddedAtomEncoder(emb_dim=atom_emb_dim)
        self.bond_encoder = PaddedBondEncoder(emb_dim=bond_emb_dim)
        self.perceiver = Perceiver(
            input_dim=2 * atom_emb_dim + 2 * node_preprocess_dim + bond_emb_dim,
            depth=p_depth,
            latent_trnsfmr_depth=p_latent_trsnfmr_depth,
            num_latents=p_num_latents,
            latent_dim=p_latent_dim,
            cross_heads=p_cross_heads,
            latent_heads=p_latent_heads,
            cross_dim_head=p_cross_dim_head,
            latent_dim_head=p_latent_dim_head,
            num_classes=2,
            attn_dropout=p_attn_dropout,
            ff_dropout=p_ff_dropout,
            weight_tie_layers=p_weight_tie_layers
        )

    def forward(self, batch_X, X_mask, device):
        """
        :param batch_X: (bs, num_nodes, num_node_feat), (bs, num_nodes, num_node_preprocess_feat),
                        (bs, num_edges, 2), (bs, num_edges, num_edge_feat)
        :param X_mask: (bs, num_nodes), (bs, num_edges)
        :param device: cuda or cpu
        """
        node_features, node_preprocess_feat, edge_index, edge_features = [X.to(device) for X in batch_X]
        node_encodings = torch.cat([self.atom_encoder(node_features), node_preprocess_feat], dim=2)

        x_1, x_2 = get_node_feature_pairs(edge_index, node_encodings, device)
        x_3 = self.bond_encoder(edge_features)
        x = torch.cat([x_1, x_2, x_3], dim=2)

        x = self.perceiver(x, mask=X_mask[1].to(device))
        return x


class HIVTransformerEncoderModel(nn.Module):
    def __init__(self, atom_emb_dim, bond_emb_dim,  node_preprocess_dim,
                 n_layers, n_heads, head_dim, pf_dim, attn_dropout, ff_dropout):
        super(HIVTransformerEncoderModel, self).__init__()
        self.atom_encoder = PaddedAtomEncoder(emb_dim=atom_emb_dim)
        self.bond_encoder = PaddedBondEncoder(emb_dim=bond_emb_dim)
        query_dim = 2 * atom_emb_dim + 2 * node_preprocess_dim + bond_emb_dim
        self.transformer_encoder = TransformerEncoder(
            query_dim=query_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            head_dim=head_dim,
            pf_dim=pf_dim,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout
        )
        self.to_logits = nn.Sequential(
            nn.LayerNorm(query_dim),
            nn.Linear(query_dim, 2)
        )

    def forward(self, batch_X, X_mask, device):
        node_features, node_preprocess_feat, edge_index, edge_features = [X.to(device) for X in batch_X]
        node_encodings = torch.cat([self.atom_encoder(node_features), node_preprocess_feat], dim=2)

        edge_mask = X_mask[1].to(device) # shape: (bs, num_edges)
        x_1, x_2 = get_node_feature_pairs(edge_index, node_encodings, device)
        x_3 = self.bond_encoder(edge_features)
        x = torch.cat([x_1, x_2, x_3], dim=2)
        x = self.transformer_encoder(x, mask=edge_mask)
        x = (x * edge_mask.unsqueeze(2)).sum(dim=1) / (edge_mask.sum(dim=1).unsqueeze(1) ** 0.5)
        return self.to_logits(x)