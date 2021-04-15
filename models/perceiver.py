from functools import wraps

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat


# helpers

def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cache_fn(f):
    cache = None

    @wraps(f)
    def cached_fn(*args, _cache=True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache

    return cached_fn


# helper classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)

        return self.fn(x, **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, hid_dim=None, dropout=0.):
        super().__init__()
        hid_dim = default(hid_dim, 4 * dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hid_dim * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, dim)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads
        self.query_dim = query_dim
        self.context_dim = context_dim

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None, mask_kv_only=True):
        h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        bs, num_q, _ = x.shape
        _, num_kv, _ = context.shape

        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask) and mask_kv_only:
            assert mask.shape == (bs, num_kv)
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, -torch.finfo(sim.dtype).max)
        elif exists(mask) and (not mask_kv_only):
            assert mask.shape == (bs, num_q, num_kv)
            mask = repeat(mask, 'b i j -> (b h) i j', h=h)
            sim.masked_fill_(~mask, -torch.finfo(sim.dtype).max)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


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

    def forward(self, data, mask):
        bs, num_tokens, _ = data.shape
        x = data

        # convert mask from shape (bs, num_tokens) to (bs, num_tokens, num_tokens)
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

    def forward(self, data, mask=None, latent_input=None):
        b = data.shape[0]

        data = rearrange(data, 'b ... d -> b (...) d')

        x = repeat(self.latents, 'n d -> b n d', b=b)
        if exists(latent_input):
            x = torch.cat([x, latent_input], dim=1)

        for cross_attn, cross_ff, latent_trnsfmr in self.layers:
            x = cross_attn(x, context=data, mask=mask) + x
            x = cross_ff(x) + x
            for latent_self_attn, latent_ff in latent_trnsfmr:
                x = latent_self_attn(x) + x
                x = latent_ff(x) + x

        return x
