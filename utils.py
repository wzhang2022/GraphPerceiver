from models.perceiver import Perceiver
from models.positional_encodings import FourierEncode
import argparse
import torch
from torch.nn.utils.rnn import pad_sequence

import traceback
import warnings
import sys


def warn_with_traceback(message, category, filename, lineno, file=None, line=None):

    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))

warnings.showwarning = warn_with_traceback


def parse_args():
    parser = argparse.ArgumentParser()
    # experiment details
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--save_file", type=str, required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--device", type=str, default="cuda")

    # architectural details
    parser.add_argument("--depth", type=int, default=3)

    # training details
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=0.0005)
    parser.add_argument("--clip", type=float, default=1)

    # data details
    return parser.parse_args()


# def make_model(args):
#     if args.model == "perceiver":
#         model = make_perceiver(args)
#     else:
#         raise Exception("invalid model type")
#     return model
#
#
# def make_perceiver(args):
#     pe_module = FourierEncode()
#     return Perceiver(
#         input_dim=args.input_dim,
#         pe_module=None,
#         num_latents = 512,
#         latent_dim = 512,
#         cross_heads = 1,
#         latent_heads = 8,
#         cross_dim_head = 64,
#         latent_dim_head = 64,
#         num_classes = 1000,
#         attn_dropout = 0.,
#         ff_dropout = 0.,
#         weight_tie_layers = False)
