import torch
import torch.nn as nn
from .helpers import PreNorm, Attention, FeedForward, exists, default, cache_fn
from einops import rearrange, repeat


class TransformerEncoder(nn.Module):
    def __init__(self, query_dim, n_layers, n_heads, head_dim, pf_dim, attn_dropout, ff_dropout):
        super(TransformerEncoder, self).__init__()
        self.query_dim = query_dim
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.layers = nn.ModuleList([
            nn.ModuleList([
                PreNorm(query_dim, Attention(query_dim, heads=n_heads, dim_head=head_dim, dropout=attn_dropout)),
                PreNorm(query_dim, FeedForward(query_dim, hid_dim=pf_dim, dropout=ff_dropout))
            ]) for _ in range(n_layers)
        ])

    def forward(self, data, mask=None):
        bs, num_tokens, _ = data.shape
        x = data

        # convert mask from shape (bs, num_tokens) to (bs, num_tokens, num_tokens)
        if exists(mask):
            mask = torch.einsum("bi,bj->bij", mask, mask)
        for self_attn, positionwise_ff in self.layers:
            x = self_attn(x, mask=mask, mask_kv_only=False) + x
            x = positionwise_ff(x) + x
        return x


# main class

class Perceiver(nn.Module):
    def __init__(
            self,
            *,
            input_dim,
            depth,
            latent_trnsfmr_depth,
            num_latents=512,
            latent_dim=512,
            cross_heads=1,
            latent_heads=8,
            cross_dim_head=64,
            latent_dim_head=64,
            attn_dropout=0.,
            ff_dropout=0.,
            weight_tie_layers=False
    ):
        super().__init__()

        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        self.num_latents = num_latents

        get_cross_attn = lambda: PreNorm(latent_dim,
                                         Attention(latent_dim, input_dim, heads=cross_heads, dim_head=cross_dim_head,
                                                   dropout=attn_dropout), context_dim=input_dim)
        get_cross_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout=ff_dropout))
        get_latent_trnsfmr = lambda: nn.ModuleList([
            nn.ModuleList([
                PreNorm(latent_dim,
                        Attention(latent_dim, heads=latent_heads, dim_head=latent_dim_head, dropout=attn_dropout)),
                PreNorm(latent_dim, FeedForward(latent_dim, dropout=ff_dropout))])
            for _ in range(latent_trnsfmr_depth)])

        get_cross_attn, get_cross_ff, get_latent_trnsfmr= map(cache_fn, (
        get_cross_attn, get_cross_ff, get_latent_trnsfmr))

        self.layers = nn.ModuleList([])
        for i in range(depth):
            should_cache = i > 0 and weight_tie_layers
            cache_args = {'_cache': should_cache}

            self.layers.append(nn.ModuleList([
                get_cross_attn(**cache_args),
                get_cross_ff(**cache_args),
                get_latent_trnsfmr(**cache_args),
            ]))

    def forward(self, data, mask=None, latent_input=None, latent_extratokens_mask=None, extratok_attn_bias_mask=None,
                bias_weights=None):
        bs, num_kv, _ = data.shape

        assert mask.shape == (bs, num_kv)

        data = rearrange(data, 'b ... d -> b (...) d')

        x = repeat(self.latents, 'n d -> b n d', b=bs)

        using_latent = exists(latent_input)
        latent_transfrmr_mask = None
        attn_bias_mask = None
        if using_latent:
            # check that number of query tokens matches with mask
            assert latent_extratokens_mask.shape[1] == latent_input.shape[1]
            x = torch.cat([x, latent_input], dim=1)

            # combine masks to get new mask of shape (bs, num_latents+num_latent_inputs, num_kv)
            latent_value_mask = torch.ones((bs, self.num_latents), dtype=bool, device=data.device)
            query_mask = torch.cat([latent_value_mask, latent_extratokens_mask], dim=1)

            latent_value_bias_mask = torch.zeros((bs, self.num_latents, num_kv), dtype=bool, device=data.device)
            attn_bias_mask = torch.cat([latent_value_bias_mask, extratok_attn_bias_mask], dim=1)
            mask = torch.einsum("bi,bj->bij", query_mask, mask)
            latent_transfrmr_mask = torch.einsum("bi,bj->bij", query_mask, query_mask)

        for i, (cross_attn, cross_ff, latent_trnsfmr) in enumerate(self.layers):
            attn_bias = None
            if exists(bias_weights):
                attn_bias = attn_bias_mask * bias_weights[i]
            x = cross_attn(x, context=data, mask=mask, mask_kv_only=not using_latent, sim_bias=attn_bias) + x
            x = cross_ff(x) + x
            for latent_self_attn, latent_ff in latent_trnsfmr:
                x = latent_self_attn(x, mask=latent_transfrmr_mask, mask_kv_only=not using_latent) + x
                x = latent_ff(x) + x

        return x
