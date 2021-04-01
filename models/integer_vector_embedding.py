import torch
from torch import nn
import torch.functional as F


class IntegerVectorEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim, num_embedding, shared=False, device=0, pad_idx=-1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        if shared:
            embedding = nn.Embedding(num_embedding, output_dim, padding_idx=pad_idx)
            self.embeddings = nn.ModuleList([embedding for _ in range(input_dim)])
        else:
            self.embeddings = nn.ModuleList([nn.Embedding(num_embedding, output_dim, padding_idx=pad_idx) for _ in range(input_dim)])

    def forward(self, int_vec):
        # int_vec = (batch_size, num_nodes, input_dim)
        bs, num_nodes, input_dim = int_vec.shape
        out = torch.zeros([bs, num_nodes, self.output_dim]).to(self.device)
        for i in range(self.input_dim):
            out += self.embeddings[i](int_vec[:, :, i])
        return out

