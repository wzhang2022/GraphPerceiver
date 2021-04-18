from einops import rearrange, reduce, repeat
import torch
from torch import einsum
import torch.nn as nn
import torch.nn.functional as F

import math
from models.perceiver import default, FeedForward


class NystromAttention(nn.Module):
    # TODO: enable arbitrary masking, thus allowing autoregressive models

    def __init__(
            self,
            query_dim,
            context_dim,
            heads,
            head_dim,
            landmarks,
            dropout):
        super().__init__()

        self.scale = head_dim ** -0.5
        self.heads = heads

        inner_dim = head_dim * heads
        self.num_landmarks = landmarks
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        """

        :param x: Tensor of shape (bs, num_q_tokens, num_heads * head_dim)
        :param context: Tensor of shape (bs, num_kv_tokens, num_heads * head_dim)
        :param mask: Tensor of shape (bs, num_q_tokens, num_kv_tokens). Each sample is rectangular; i.e. for a sample
        with n queries and m kv-pairs, the upper left n x m block is all True, everywhere else is False
        :return:
        """
        context = default(context, x)
        # get x and context masks
        q_mask = mask.any(dim=2)        # shape (bs, num_q)
        kv_mask = mask.any(dim=1)       # shape (bs, num_kv)

        # pad number of input tokens if not divisible
        assert x.shape[1] % self.num_landmarks == 0
        assert context.shape[1] % self.num_landmarks == 0
        q = self.to_q(x) * (self.scale ** 0.5) * q_mask.unsqueeze(2)
        k, v = (self.to_kv(context) * (self.scale ** 0.5) * kv_mask.unsqueeze(2)).chunk(2, dim=-1)
        h = self.heads
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))          # compute for each head
        q_landmarks, k_landmarks = map(lambda t: reduce(t, 'b (l n) d -> b l d', "mean", l=self.num_landmarks), (q, k))
        kernel_1 = F.softmax(einsum("b i d, b j d -> b i j", q, k_landmarks), dim=-1)            # (bs * h, num_q, l)
        kernel_2 = F.softmax(einsum("b i d, b j d -> b i j", q_landmarks, k_landmarks), dim=-1)  # (bs * h, l, l)
        sim_ql_k = einsum("b i d, b j d -> b i j", q_landmarks, k)                               # (bs * h, l, num_k)
        softmax_mask = repeat(kv_mask, "b n -> (b h) () n", h=h)
        kernel_3 = F.softmax(sim_ql_k - softmax_mask * torch.finfo(sim_ql_k.dtype).max, dim=-1)

        # multiply: kernel_1 * pinverse(kernel_2) * kernel_3 * v
        x = torch.matmul(torch.matmul(kernel_1, self.iterative_inv(kernel_2)), torch.matmul(kernel_3, v))

        x = rearrange(x, '(b h) n d -> b n (h d)', h=h)

        return self.to_out(x)

    def iterative_inv(self, mat, n_iter=8):
        """

        :param mat: tensor of shape (bs, l, l)
        :param n_iter: number of iterations for approximation
        :return: pseudoinverse approximation of mat
        """
        id = torch.eye(mat.size(-1), device=mat.device)
        k = mat
        scale_1 = 1 / torch.max(torch.sum(torch.abs(k), dim=-2), dim=-1)[0]    # shape: (bs,)
        scale_2 = 1 / torch.max(torch.sum(torch.abs(k), dim=-1), dim=-1)[0]    # shape: (bs,)
        v = k.transpose(-1, -2) * scale_1[:, None, None] * scale_2[:, None, None]
        for _ in range(n_iter):
            kv = torch.matmul(k, v)
            v = torch.matmul(0.25 * v, 13 * id - torch.matmul(kv, 15 * id - torch.matmul(kv, 7 * id - kv)))
        return v


class Nystromformer(nn.Module):
    def __init__(self, query_dim, n_layers, n_heads, head_dim, n_landmarks, pf_dim, attn_dropout, ff_dropout):
        super(Nystromformer, self).__init__()
        self.query_dim = query_dim
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.layers = nn.ModuleList([
            nn.ModuleList([
                nn.LayerNorm(query_dim),
                NystromAttention(
                    query_dim=query_dim,
                    context_dim=query_dim,
                    heads=n_heads,
                    head_dim=head_dim,
                    landmarks=n_landmarks,
                    dropout=attn_dropout),
                nn.LayerNorm(query_dim),
                FeedForward(query_dim, hid_dim=pf_dim, dropout=ff_dropout)
            ]) for _ in range(n_layers)
        ])
        self.landmarks = n_landmarks

    def forward(self, data, mask):
        bs, num_tokens, query_dim = data.shape
        padded_num_tokens = math.ceil(num_tokens / self.landmarks) * self.landmarks

        x = torch.zeros((bs, padded_num_tokens, query_dim), dtype=data.dtype, device=data.device)
        padded_mask = torch.zeros((bs, padded_num_tokens), dtype=bool, device=mask.device)
        x[:, :num_tokens, :] = data
        padded_mask[:, :num_tokens] = mask

        # convert mask from shape (bs, num_tokens) to (bs, num_tokens, num_tokens)
        mask = torch.einsum("bi,bj->bij", padded_mask, padded_mask)
        for norm1, self_attn, norm2, positionwise_ff in self.layers:
            x = self_attn(norm1(x), mask=mask) + x
            x = positionwise_ff(norm2(x)) + x
        return x[:, :num_tokens, :]
