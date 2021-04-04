import torch.nn as nn
from models.perceiver import Perceiver
from models.padded_mol_encoder import PaddedAtomEncoder


class HIVModel(nn.Module):
    def __init__(self, atom_emb_dim, perceiver_depth):
        super(HIVModel, self).__init__()
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

    def forward(self, x, mask):
        x = self.atom_encoder(x)
        x = self.perceiver(x, mask=mask)
        return x