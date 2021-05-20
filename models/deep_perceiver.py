from torch import einsum
import torch.nn as nn
from einops import rearrange, repeat
from .helpers import cache_fn, PreNorm, FeedForward, Attention


class TransformerBlock(nn.Module):
    def __init__(self, query_dim, kv_dim, n_heads, head_dim, pf_dim, attn_dropout, ff_dropout):
        super(TransformerBlock, self).__init__()
        self.query_dim = query_dim
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.attn = PreNorm(query_dim, Attention(query_dim, kv_dim, heads=n_heads, dim_head=head_dim, dropout=attn_dropout))
        self.pos_ff = PreNorm(query_dim, FeedForward(query_dim, hid_dim=pf_dim, dropout=ff_dropout))


    def forward(self, data_1, data_2, mask):
        bs, num_data1_toks, _ = data_1.shape
        _, num_data2_toks, _ = data_2.shape
        assert mask.shape == (bs, num_data1_toks, num_data2_toks)
        x = data_1
        x = self.attn(data_1, context=data_2, mask=mask, mask_kv_only=False) + x
        x = self.pos_ff(x) + x
        return x


class DeepPerceiver(nn.Module):
    def __init__(
            self,
            input_dim_1,
            input_dim_2,
            depth,
            heads=8,
            latent_dim_head=64,
            attn_dropout=0.,
            ff_dropout=0.,
            weight_tie_layers=False):
        super(DeepPerceiver, self).__init__()
        self.input_dim_1 = input_dim_1
        self.input_dim_2 = input_dim_2
        get_transformer_block_1 = lambda: TransformerBlock(input_dim_1, input_dim_2, heads, latent_dim_head,
                                                           input_dim_1 * 4, attn_dropout, ff_dropout)
        get_transformer_block_2 = lambda: TransformerBlock(input_dim_2, input_dim_1, heads, latent_dim_head,
                                                           input_dim_2 * 4, attn_dropout, ff_dropout)
        get_transformer_block_1, get_transformer_block_2 = map(cache_fn, (get_transformer_block_1,
                                                                          get_transformer_block_2))
        self.layers = nn.ModuleList([])
        for i in range(depth):
            should_cache = i > 0 and weight_tie_layers
            cache_args = {'_cache': should_cache}

            self.layers.append(
                nn.ModuleList([
                    get_transformer_block_1(**cache_args),
                    get_transformer_block_2(**cache_args),
                ])
            )

    def forward(self, data1, data2, mask1, mask2):
        assert data1.shape[2] == self.input_dim_1 and data2.shape[2] == self.input_dim_2
        mask = einsum("bi,bj->bij", mask1, mask2)
        x, y = data1, data2
        for block1, block2 in self.layers:
            x, y = block1(x, y, mask), block2(y, x, rearrange(mask, "b x y -> b y x"))
        return x
