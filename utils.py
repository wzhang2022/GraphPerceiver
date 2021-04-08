import argparse
import torch

import traceback
import warnings
import sys
import random
import numpy as np

from scipy.sparse.linalg import eigsh

from torch.nn.utils.rnn import pad_sequence
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims

# set random seeds
SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


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


def hiv_graph_collate(batch):
    """

    :param batch: List[{'edge_index': np.array(2, num_edges), 'edge_feat': np.array(num_edges, num_edge_feat),
                        'node_feat': np.array(num_nodes, num_node_feat), 'num_nodes': num_nodes}]
    :return: graph description as tensors, graph masks, graph labels
    """
    full_atom_feature_dims = get_atom_feature_dims()
    full_bond_feature_dims = get_bond_feature_dims()
    node_features, node_mask = variable_pad_sequence([torch.as_tensor(item[0]['node_feat']) for item in batch],
                                                     full_atom_feature_dims)
    edge_features, edge_mask = variable_pad_sequence([torch.as_tensor(item[0]['edge_feat']) for item in batch],
                                                     full_bond_feature_dims)
    edge_index = pad_sequence([torch.as_tensor(item[0]['edge_index'].transpose()) for item in batch], batch_first=True,
                              padding_value=0)
    labels = torch.Tensor([item[1][0] for item in batch]).long()
    return (node_features, edge_index, edge_features), (node_mask, edge_mask), labels


def variable_pad_sequence(sequences, pad_idxs):
    """

    :param sequences: list of batch_size sequences of tensors of shape (num_tokens, num_features)
    :param pad_idxs: list of num_features length, where we pad with a different length for each feature
    :return: feature tensor (batch_size, max_num_tokens, num_features), mask tensor (batch_size, max_num_tokens)
    """
    batch_size = len(sequences)
    max_num_tokens = max([seq.shape[0] for seq in sequences])
    num_features = sequences[0].shape[1]
    output = sequences[0].new_full((batch_size, max_num_tokens, num_features), 0)
    mask = torch.as_tensor(np.zeros((batch_size, max_num_tokens), dtype=np.bool))
    for i in range(batch_size):
        output[i] = torch.as_tensor(pad_idxs)
        seq = sequences[i]
        output[i, :len(seq)] = seq
        mask[i, :len(seq)] = True
    return output, mask


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


### Laplacian Embeddings Utils

def get_LPE_embeddings(n_nodes, edges, k):
    """
    :n_nodes: |V|
    :edges: [list of head nodes, list of tail nodes]
    :k: hyperparameter, determines how many eigenvectors to use for PE
    """
    A = np.zeros(shape=(n_nodes, n_nodes))           # initialize (negative) adjacency matrix
    for j in range(len(edges[0])):
        A[edges[0][j], edges[1][j]] -= 1             # [row (0), column (1)]
    
    D_sqrinv = np.zeros(shape=(n_nodes, n_nodes))    # initialize D^{-1/2}
    for j in range(len(A[0])):
        D_sqrinv[j,j] = sum(-1 * A[j]) ** (-0.5)
    
    L = D_sqrinv @ (A) @ D_sqrinv                     # (normalized) Laplacian
    for j in range(len(A[0])):
        L[j,j] += 1                                   # adding I
        
    return eigsh(L, k)[1]                             # returns eigenvectors only (no values)

# def get_evectors(laplacian, k):
#     ans = eigsh(laplacian, k)
#     return ans[1]

