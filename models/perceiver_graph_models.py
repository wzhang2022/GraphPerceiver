import torch
import torch.nn as nn
from models.perceiver import Perceiver, TransformerEncoder
from models.nystromformer import Nystromformer
from models.padded_mol_encoder import PaddedAtomEncoder, PaddedBondEncoder
from einops import rearrange
from einops.layers.torch import Reduce


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
            attn_dropout=0.,
            ff_dropout=0.,
            weight_tie_layers=False
        )

    def forward(self, batch_X, X_mask, device):
        node_features, edge_index, edge_features = batch_X
        x = self.atom_encoder(node_features.to(device))
        x = self.perceiver(x, mask=X_mask[0].to(device))
        return x


class MoleculePerceiverModel(nn.Module):
    def __init__(self, atom_emb_dim, bond_emb_dim, node_preprocess_dim,
                 p_depth, p_latent_trsnfmr_depth, p_num_latents, p_latent_dim, p_cross_heads, p_latent_heads,
                 p_cross_dim_head, p_latent_dim_head, p_attn_dropout, p_ff_dropout, p_weight_tie_layers,
                 p_node_edge_cross_attn, p_num_outputs, connection_bias, multi_classification, num_classifiers,
                 classifier_transformer_layers):
        super(MoleculePerceiverModel, self).__init__()
        self.atom_encoder = PaddedAtomEncoder(emb_dim=atom_emb_dim)
        self.bond_encoder = PaddedBondEncoder(emb_dim=bond_emb_dim)
        self.latent_atom_encode = PaddedAtomEncoder(emb_dim=p_latent_dim)
        self.latent_dim = p_latent_dim
        self.node_edge_cross_attn = p_node_edge_cross_attn
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
            attn_dropout=p_attn_dropout,
            ff_dropout=p_ff_dropout,
            weight_tie_layers=p_weight_tie_layers
        )

        self.to_logits = nn.Sequential(
            nn.LayerNorm(p_latent_dim),
            nn.Linear(p_latent_dim, p_num_outputs)
        )

        self.connection_bias = connection_bias
        if connection_bias:
            self.connection_bias_weight = None

        self.multi_classification = multi_classification
        if multi_classification:
            assert p_num_outputs % num_classifiers == 0
            self.classifiers = nn.ModuleList([
                nn.Sequential(
                    TransformerEncoder(
                        query_dim=p_latent_dim,
                        n_layers=classifier_transformer_layers,
                        n_heads=p_latent_heads,
                        head_dim=p_latent_dim_head,
                        pf_dim=p_latent_dim * 2,
                        attn_dropout=p_attn_dropout,
                        ff_dropout=p_attn_dropout),
                    Reduce("b n d -> b d", "mean"),
                    nn.LayerNorm(p_latent_dim),
                    nn.Linear(p_latent_dim, p_num_outputs // num_classifiers)
                )
                for _ in range(num_classifiers)])

        self.to_logits = nn.Sequential(
            Reduce("b n d -> b d", "mean"),
            nn.LayerNorm(p_latent_dim),
            nn.Linear(p_latent_dim, p_num_outputs)
        )

    def forward(self, batch_X, X_mask, device, node_pert=None):
        """
        :param batch_X: (bs, num_nodes, num_node_feat), (bs, num_nodes, num_node_preprocess_feat),
                        (bs, num_edges, 2), (bs, num_edges, num_edge_feat)
        :param X_mask: (bs, num_nodes), (bs, num_edges)
        :param device: cuda or cpu
        """
        node_features, node_preprocess_feat, edge_index, edge_features = [X.to(device) for X in batch_X]
        node_encodings = torch.cat([self.atom_encoder(node_features), node_preprocess_feat], dim=2)
        if node_pert is not None:
            node_encodings += node_pert

        x_1, x_2 = get_node_feature_pairs(edge_index, node_encodings, device)
        x_3 = self.bond_encoder(edge_features)
        x = torch.cat([x_1, x_2, x_3], dim=2)

        if self.node_edge_cross_attn:
            latent_node_feat = self.latent_atom_encode(node_features)
            x = self.perceiver(x, mask=X_mask[1].to(device), latent_input=latent_node_feat)
            x = x[:, :self.latent_dim, :]
        else:
            x = self.perceiver(x, mask=X_mask[1].to(device))

        if self.multi_classification:
            return torch.cat([cls(x) for cls in self.classifiers], dim=-1)
        else:
            return self.to_logits(x)


class PCBAtoHIVPerceiverTransferModel(nn.Module):
    def __init__(self, pretrained_model: MoleculePerceiverModel, layers_to_unfreeze, epochs_before_unfreeze, lr_for_unfrozen):
        super(PCBAtoHIVPerceiverTransferModel, self).__init__()

        # freeze pretrained_model parameters
        for params in pretrained_model.parameters():
            params.requires_grad = False

        self.layers_to_unfreeze = layers_to_unfreeze
        self.epochs_before_unfreeze = epochs_before_unfreeze
        self.atom_encoder = pretrained_model.atom_encoder
        self.bond_encoder = pretrained_model.bond_encoder
        self.latent_atom_encode = pretrained_model.latent_atom_encode
        self.perceiver = pretrained_model.perceiver
        self.latent_dim = pretrained_model.latent_dim
        self.pretrained_model = pretrained_model
        self.lr_for_unfrozen = lr_for_unfrozen

        # make trainable final layers
        self.to_logits = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, 2)
        )

    def unfreeze_layers(self, optimizer):
        for params in list(self.pretrained_model.parameters())[-self.layers_to_unfreeze:]:
            params.requires_grad = True

        for g in optimizer.param_groups:
            g['lr'] = self.lr_for_unfrozen

    def forward(self, batch_X, X_mask, device):
        node_features, node_preprocess_feat, edge_index, edge_features = [X.to(device) for X in batch_X]
        node_encodings = torch.cat([self.atom_encoder(node_features), node_preprocess_feat], dim=2)

        x_1, x_2 = get_node_feature_pairs(edge_index, node_encodings, device)
        x_3 = self.bond_encoder(edge_features)
        x = torch.cat([x_1, x_2, x_3], dim=2)
        x = self.perceiver(x, mask=X_mask[1].to(device))
        x = x.mean(dim=-2)
        return self.to_logits(x)


class DomainAdapter(nn.Module):
    def __init__(self, encoder, discriminator, classifier):
        super(DomainAdapter, self).__init__()
        self.encoder = encoder
        self.discriminator = discriminator
        self.classifier = classifier

    def forward(self, *args):
        """
        :param batch_X: (bs, num_nodes, num_node_feat), (bs, num_nodes, num_node_preprocess_feat),
                        (bs, num_edges, 2), (bs, num_edges, num_edge_feat)
        :param X_mask: (bs, num_nodes), (bs, num_edges)
        :param device: cuda or cpu
        """
        # TODO: fill in domain adaptation
        raise NotImplementedError


class MoleculeTransformerEncoderModel(nn.Module):
    def __init__(self, atom_emb_dim, bond_emb_dim, node_preprocess_dim,
                 n_layers, n_heads, head_dim, pf_dim, attn_dropout, ff_dropout, num_outputs,
                 nystrom=False, n_landmarks=32):
        super(MoleculeTransformerEncoderModel, self).__init__()
        self.atom_encoder = PaddedAtomEncoder(emb_dim=atom_emb_dim)
        self.bond_encoder = PaddedBondEncoder(emb_dim=bond_emb_dim)
        query_dim = 2 * atom_emb_dim + 2 * node_preprocess_dim + bond_emb_dim
        if nystrom:
            self.transformer_encoder = Nystromformer(
                query_dim=query_dim,
                n_layers=n_layers,
                n_heads=n_heads,
                n_landmarks=n_landmarks,
                head_dim=head_dim,
                pf_dim=pf_dim,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
            )
        else:
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
            nn.Linear(query_dim, num_outputs)
        )

    def forward(self, batch_X, X_mask, device):
        node_features, node_preprocess_feat, edge_index, edge_features = [X.to(device) for X in batch_X]
        node_encodings = torch.cat([self.atom_encoder(node_features), node_preprocess_feat], dim=2)

        edge_mask = X_mask[1].to(device)  # shape: (bs, num_edges)
        x_1, x_2 = get_node_feature_pairs(edge_index, node_encodings, device)
        x_3 = self.bond_encoder(edge_features)
        x = torch.cat([x_1, x_2, x_3], dim=2)
        x = self.transformer_encoder(x, mask=edge_mask)
        x = (x * edge_mask.unsqueeze(2)).sum(dim=1)
        return self.to_logits(x)
