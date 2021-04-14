import argparse
import torch

import traceback
import warnings
import sys
import random
import numpy as np

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
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
    parser.add_argument("--latent_transformer_depth", type=int, default=4)
    parser.add_argument("--num_latents", type=int, default=64)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--cross_heads", type=int, default=1)
    parser.add_argument("--latent_heads", type=int, default=8)
    parser.add_argument("--cross_dim_head", type=int, default=32)
    parser.add_argument("--latent_dim_head", type=int, default=32)
    parser.add_argument("--attn_dropout", type=float, default=0.0)
    parser.add_argument("--ff_dropout", type=float, default=0.0)
    parser.add_argument("--weight_tie_layers", type=bool, default=False)

    
    # embedding details
    parser.add_argument("--atom_emb_dim", type=int, required=True)
    parser.add_argument("--bond_emb_dim", type=int, required=True)
    parser.add_argument("--k_eigs", type=int, required=True)      # specifies # of e-vectors to use

    # training details
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=0.0005)
    parser.add_argument("--lr_decay", type=float, default=1)               # gamma, i.e. lr * gamma when scheduler.step() is called
    parser.add_argument("--scheduler", type=str, default='exponential')    # 'exponential', 'multistep', 'plateau'
    parser.add_argument("--milestone_frequency", type=int, default=10)     # how far apart the milestones are
    parser.add_argument("--milestone_start", type=int, default=1)          # first epoch in the milestones array
    parser.add_argument("--milestone_end", type=int, default=1)            # last epoch in the milestones array
    parser.add_argument("--clip", type=float, default=1)

    # data details
    return parser.parse_args()


class GraphDataset(Dataset):
    def __init__(self, graphs, preprocess=[]):
        """
        :param graphs: list of dictionaries
        :param preprocess: list of preprocessing functions that are applied to each dictionary, left-to-right
        """
        self.graphs = graphs
        for i in range(len(self.graphs)):
            for preprocess_fn in preprocess:
                self.graphs[i] = preprocess_fn(self.graphs[i])
            if not preprocess:
                num_nodes = self.graphs[i][0]["num_nodes"]
                self.graphs[i][0]["node_preprocess_feat"] = np.zeros((num_nodes, 0))

    def __getitem__(self, idx):
        return self.graphs[idx]

    def __len__(self):
        return len(self.graphs)


def hiv_graph_collate(batch):
    """

    :param batch: List[{'edge_index': np.array(2, num_edges), 'node_preprocess_feat': np.array(num_nodes, num_node_preprocess_feat),
                        'edge_feat': np.array(num_edges, num_edge_feat), 'node_feat': np.array(num_nodes, num_node_feat), 'num_nodes': num_nodes}]
    :return: graph description as tensors, graph masks, graph labels
    """
    full_atom_feature_dims = get_atom_feature_dims()
    full_bond_feature_dims = get_bond_feature_dims()
    node_feat, node_mask = variable_pad_sequence([torch.as_tensor(item[0]['node_feat']) for item in batch],
                                                 full_atom_feature_dims)
    node_preprocess_feat = pad_sequence([torch.as_tensor(item[0]['node_preprocess_feat'], dtype=torch.float32) for item in batch],
                                        batch_first=True, padding_value=0)
    edge_feat, edge_mask = variable_pad_sequence([torch.as_tensor(item[0]['edge_feat']) for item in batch],
                                                 full_bond_feature_dims)
    edge_index = pad_sequence([torch.as_tensor(item[0]['edge_index'].transpose()) for item in batch], batch_first=True,
                              padding_value=0)
    labels = torch.Tensor([item[1][0] for item in batch]).long()
    return (node_feat, node_preprocess_feat, edge_index, edge_feat), (node_mask, edge_mask), labels


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
        output[i] = torch.as_tensor(pad_idxs).unsqueeze(0)
        seq = sequences[i]
        output[i, :len(seq)] = seq
        mask[i, :len(seq)] = True
    return output, mask


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Laplacian Embeddings Utils

class LPE(object):
    """
    Given raw data from GraphPropPredDataset, add first k eigenvectors
    Args:
        k (int): Desired number of eigenvectors for PE
    """
    def __init__(self, k):
        self.k = k
        
    def __call__(self, data_sample):
        dictionary = data_sample[0]
        y = data_sample[1]
        
        # keys: ['edge_index', 'edge_feat', 'node_feat', 'num_nodes']
        positional_embeddings = get_LPE_embeddings(dictionary['num_nodes'], dictionary['edge_index'], self.k)
        
        new_dictionary = {'edge_index': dictionary['edge_index'],
                          'edge_feat': dictionary['edge_feat'],
                          'node_feat': dictionary['node_feat'],
                          'num_nodes': dictionary['num_nodes'],
                          'node_preprocess_feat': positional_embeddings}
        
        return (new_dictionary, y)
    

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
        deg = sum(A[j])                              # this is nonpositive
        if deg == 0:
            D_sqrinv[j,j] = 0
        else:
            D_sqrinv[j,j] = (-1 * deg) ** (-0.5)
    
    L = D_sqrinv @ (A) @ D_sqrinv                     # (normalized) Laplacian
    for j in range(len(A[0])):
        L[j,j] += 1                                   # adding I
    
    # eigh returns evalues [0] and evectors [1] in ascending order 
    vecs = np.linalg.eigh(L)[1]
    
    if k > n_nodes:
        zero_vecs = np.zeros(shape=((k-n_nodes), n_nodes))
        eigvecs = np.concatenate((zero_vecs.T, vecs), axis=1)    # pad with 0 vectors
        assert eigvecs.shape == (n_nodes, k)
        return eigvecs
    
    elif k == n_nodes:
        assert vecs.shape == (n_nodes, k)
        return vecs
    else:
        eigvecs = vecs[:, n_nodes-k:]            # k columns corresponding to greatest eigenvalues   
        assert eigvecs.shape == (n_nodes, k)
        return eigvecs


