from math import pi, log
import torch
import torch.nn as nn
from einops import repeat, rearrange


def fourier_encode(x, max_freq, num_freq_bands, base):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    scales = torch.logspace(1., log(max_freq / 2) / log(base), num_freq_bands, base=base,
                            device=device, dtype=dtype)
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    x = x * scales * pi
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim=-1)
    return x


class FourierEncode(nn.Module):
    def __init__(self, max_freq=2048, num_bands=4, base=2):
        self.max_freq = max_freq
        self.num_bands = num_bands
        self.base = base

    def forward(self, data):
        b, *axis, _, device = data.shape, data.device
        # calculate fourier encoded positions in the range of [-1, 1], for all axis

        axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps=size, device=device), axis))
        pos = torch.stack(torch.meshgrid(*axis_pos), dim = -1)
        enc_pos = fourier_encode(pos, self.max_freq, self.num_freq_bands, base = self.base)
        enc_pos = rearrange(enc_pos, '... n d -> ... (n d)')
        enc_pos = repeat(enc_pos, '... -> b ...', b = b)
        return enc_pos

