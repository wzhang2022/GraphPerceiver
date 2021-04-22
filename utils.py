import argparse
import torch

import traceback
import warnings
import sys
import numpy as np

import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims
from pytorch_lamb import Lamb
from ogb.graphproppred import GraphPropPredDataset
from functools import partial
from torch.utils.data import DataLoader

from models.perceiver_graph_models import MoleculePerceiverModel, MoleculeTransformerEncoderModel,\
    PCBAtoHIVPerceiverTransferModel
from models.loss_functions import CombinedCEandAUCLoss, SoftAUC, MultitaskCrossEntropyLoss


def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file, 'write') else sys.stderr
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
    parser.add_argument("--dataset", type=str, required=True)  # molhiv, molpcba
    parser.add_argument("--shuffle_split", dest="shuffle_split", action="store_true")
    parser.set_defaults(shuffle_split=False)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--load", type=str)
    parser.add_argument("--load_epoch_start", type=int, default=0)
    parser.add_argument("--transfer_learn", dest="transfer_learn", action="store_true")
    parser.set_defaults(transfer_learn=False)
    parser.add_argument("--num_flag_steps", type=int, default=0)
    parser.add_argument("--flag_step_size", type=float, default=0.001)

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
    parser.add_argument("--node_edge_cross_attn", dest="node_edge_cross_attn", action="store_true")
    parser.set_defaults(node_edge_cross_attn=False)
    parser.add_argument("--nystrom", dest="nystrom", action="store_true")
    parser.set_defaults(nystrom=False)
    parser.add_argument("--landmarks", type=int, default=32)
    parser.add_argument("--multi_classifier", dest="multi_classifier", action="store_true")
    parser.set_defaults(multi_classifier=False)
    parser.add_argument("--num_classifier", type=int, default=1)

    # embedding details
    parser.add_argument("--atom_emb_dim", type=int, required=True)
    parser.add_argument("--bond_emb_dim", type=int, required=True)
    parser.add_argument("--k_eigs", type=int, required=True)  # specifies # of e-vectors to use

    # training details
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=0.0005)
    parser.add_argument("--lr_decay", type=float, default=1.0)  # gamma, i.e. lr * gamma when scheduler.step() is called
    parser.add_argument("--scheduler", type=str, default='exponential')  # 'exponential', 'multistep', 'plateau'
    parser.add_argument("--milestone_frequency", type=int, default=10)  # how far apart the milestones are
    parser.add_argument("--milestone_start", type=int, default=1)  # first epoch in the milestones array
    parser.add_argument("--milestone_end", type=int, default=1)  # last epoch in the milestones array
    parser.add_argument("--clip", type=float, default=1)

    parser.add_argument("--optimizer", type=str, default='SGD')  # 'SGD', 'Adam', 'AdamW', 'AMSGrad', LAMB'
    parser.add_argument("--Adam_weight_decay", type=float, default=0.0)  # used for Adam and LAMB
    parser.add_argument("--Adam_beta_1", type=float, default=0.9)  # used for Adam and LAMB
    parser.add_argument("--Adam_beta_2", type=float, default=0.999)  # used for Adam and LAMB

    parser.add_argument("--criterion", type=str, default="ce")  # ce, soft_auc, combined_ce_auc
    parser.add_argument("--ce_weighted", dest="ce_weighted", action="store_true")
    parser.add_argument("--ce_unweighted", dest="ce_weighted", action="store_false")
    parser.set_defaults(ce_weighted=True)
    parser.add_argument("--auc_weight", type=float, default=1.0)

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


def mol_graph_collate(batch, dataset_name):
    """

    :param batch: List[{'edge_index': np.array(2, num_edges), 'node_preprocess_feat': np.array(num_nodes, num_node_preprocess_feat),
                        'edge_feat': np.array(num_edges, num_edge_feat), 'node_feat': np.array(num_nodes, num_node_feat), 'num_nodes': num_nodes}]
    :return: graph description as tensors, graph masks, graph labels
    """
    full_atom_feature_dims = get_atom_feature_dims()
    full_bond_feature_dims = get_bond_feature_dims()
    node_feat, node_mask = variable_pad_sequence([torch.as_tensor(item[0]['node_feat']) for item in batch],
                                                 full_atom_feature_dims)
    node_preprocess_feat = pad_sequence(
        [torch.as_tensor(item[0]['node_preprocess_feat'], dtype=torch.float32) for item in batch],
        batch_first=True, padding_value=0)
    edge_feat, edge_mask = variable_pad_sequence([torch.as_tensor(item[0]['edge_feat']) for item in batch],
                                                 full_bond_feature_dims)
    edge_index = pad_sequence([torch.as_tensor(item[0]['edge_index'].transpose()) for item in batch], batch_first=True,
                              padding_value=0)
    if dataset_name == "molhiv":
        labels = torch.Tensor([item[1][0] for item in batch]).long()
    elif dataset_name == "molpcba":
        labels = torch.Tensor([item[1] for item in batch])
    else:
        raise Exception("Dataset collation not implemented yet")
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
    A = np.zeros(shape=(n_nodes, n_nodes))  # initialize (negative) adjacency matrix
    for j in range(len(edges[0])):
        A[edges[0][j], edges[1][j]] -= 1  # [row (0), column (1)]

    D_sqrinv = np.zeros(shape=(n_nodes, n_nodes))  # initialize D^{-1/2}
    for j in range(len(A[0])):
        deg = sum(A[j])  # this is nonpositive
        if deg == 0:
            D_sqrinv[j, j] = 0
        else:
            D_sqrinv[j, j] = (-1 * deg) ** (-0.5)

    L = D_sqrinv @ (A) @ D_sqrinv  # (normalized) Laplacian
    for j in range(len(A[0])):
        L[j, j] += 1  # adding I

    # eigh returns evalues [0] and evectors [1] in ascending order 
    vecs = np.linalg.eigh(L)[1]

    if k > n_nodes:
        zero_vecs = np.zeros(shape=((k - n_nodes), n_nodes))
        eigvecs = np.concatenate((zero_vecs.T, vecs), axis=1)  # pad with 0 vectors
        assert eigvecs.shape == (n_nodes, k)
        return eigvecs

    elif k == n_nodes:
        assert vecs.shape == (n_nodes, k)
        return vecs
    else:
        eigvecs = vecs[:, n_nodes - k:]  # k columns corresponding to greatest eigenvalues
        assert eigvecs.shape == (n_nodes, k)
        return eigvecs


# factory methods for getting the components for training

def make_model(args):
    num_outputs_dict = {"molhiv": 2, "molpcba": 128}
    model_dataset = "molpcba" if args.transfer_learn else args.dataset
    if args.model == "perceiver":
        model = MoleculePerceiverModel(atom_emb_dim=args.atom_emb_dim, bond_emb_dim=args.bond_emb_dim,
                                       node_preprocess_dim=args.k_eigs,
                                       p_depth=args.depth, p_latent_trsnfmr_depth=args.latent_transformer_depth,
                                       p_num_latents=args.num_latents, p_latent_dim=args.latent_dim,
                                       p_cross_heads=args.cross_heads, p_latent_heads=args.latent_heads,
                                       p_cross_dim_head=args.cross_dim_head, p_latent_dim_head=args.latent_dim_head,
                                       p_attn_dropout=args.attn_dropout, p_ff_dropout=args.ff_dropout,
                                       p_weight_tie_layers=args.weight_tie_layers,
                                       p_node_edge_cross_attn=args.node_edge_cross_attn,
                                       p_num_outputs=num_outputs_dict[model_dataset], connection_bias=False,
                                       multi_classification=args.multi_classifier, num_classifiers=args.num_classifier)
    elif args.model == "transformer":
        model = MoleculeTransformerEncoderModel(atom_emb_dim=args.atom_emb_dim, bond_emb_dim=args.bond_emb_dim,
                                                node_preprocess_dim=args.k_eigs,
                                                n_layers=args.latent_transformer_depth, n_heads=args.latent_heads,
                                                head_dim=args.latent_dim_head, pf_dim=None,
                                                attn_dropout=args.attn_dropout, ff_dropout=args.ff_dropout,
                                                num_outputs=num_outputs_dict[model_dataset],
                                                nystrom=args.nystrom, n_landmarks=args.landmarks)
    else:
        raise Exception("invalid model type")
    if args.load is not None:
        print("loading model")
        model.load_state_dict(torch.load(f"{args.load}.pt"))
    if args.transfer_learn:
        model = PCBAtoHIVPerceiverTransferModel(model)

    return model



def make_criterion(args):
    if args.dataset == "molhiv":
        # predictions: (bs, 2), labels: (bs,)
        if args.criterion == "ce" and args.ce_weighted:
            return nn.CrossEntropyLoss(reduction="mean", weight=torch.as_tensor([1232 / 32901, 1]))
        elif args.criterion == "ce" and (not args.ce_weighted):
            return nn.CrossEntropyLoss(reduce="mean")
        elif args.criterion == "soft_auc":
            return SoftAUC()
        elif args.criterion == "combined_ce_auc":
            return CombinedCEandAUCLoss(ce_weighted=args.ce_weighted, auc_weight=args.auc_weight)
        else:
            raise Exception("Invalid training criterion")
    elif args.dataset == "molpcba":
        # predictions: (bs, 128), labels: (bs, 128)
        if args.criterion == "ce":
            return MultitaskCrossEntropyLoss(num_tasks=128)
        else:
            raise Exception("Invalid training criterion")
    else:
        raise Exception("Dataset criterion not implemented yet")


def make_optimizer(args, model):
    optimizer_type = args.optimizer

    if optimizer_type == 'SGD':
        return torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    elif optimizer_type == 'Adam':
        beta_tuple = (args.Adam_beta_1, args.Adam_beta_2)
        return torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=beta_tuple,
                                weight_decay=args.Adam_weight_decay)
    elif optimizer_type == 'AdamW':
        beta_tuple = (args.Adam_beta_1, args.Adam_beta_2)
        return torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=beta_tuple,
                                 weight_decay=args.Adam_weight_decay)
    elif optimizer_type == 'AMSGrad':
        beta_tuple = (args.Adam_beta_1, args.Adam_beta_2)
        return torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=beta_tuple,
                                 weight_decay=args.Adam_weight_decay, amsgrad=True)
    elif optimizer_type == 'LAMB':
        beta_tuple = (args.Adam_beta_1, args.Adam_beta_2)
        return Lamb(model.parameters(), lr=args.learning_rate, betas=beta_tuple,
                    weight_decay=args.Adam_weight_decay, )
    else:
        raise Exception("Invalid optimizer provided")


def make_scheduler(args, optimizer):
    scheduler_type = args.scheduler

    gamma = args.lr_decay  # not the same as Adam_weight_decay; when Adam_weight_decay > 0 we should use ExponentialLR(., 1)
    if scheduler_type == 'exponential':
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)

    elif scheduler_type == 'multistep':
        default_milestones = [84, 102, 114]  # for non-regular milestones, hardcode ahead of time

        m_freq = args.milestone_frequency
        m_start = args.milestone_start
        m_end = args.milestone_end

        # activate default_milestones when m_freq = 0
        if m_freq < 1:
            milestones = default_milestones
        else:
            assert m_end >= m_start
            milestones = list(range(m_start, min(args.n_epochs, m_end + 1), m_freq))  # [m_start, m_start+m_freq, ...]
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma)

    elif scheduler_type == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=gamma, patience=5)
        # TODO: choose parameters for threshold, epsilon, min_lr, and cooldown
        # these parameters are more relevant when we see more stability in end-game training loss

    else:
        raise Exception("Invalid scheduler provided")


def make_dataloaders(args):
    dataset = GraphPropPredDataset(name=f"ogbg-{args.dataset}", root='dataset/')
    if args.shuffle_split:
        print("Shuffled data split")
        indices = np.random.permutation(len(dataset))
        idx1, idx2 = int(len(dataset) * 0.8), int(len(dataset) * 0.9)
        train, valid, test = indices[:idx1], indices[idx1:idx2], indices[idx2:]
        split_idx = {"train": train, "valid": valid, "test": test}
    else:
        print("Original data split")
        split_idx = dataset.get_idx_split()

    graph_preprocess_fns = [LPE(args.k_eigs)] if args.k_eigs > 0 else []
    train_data = GraphDataset([dataset[i] for i in split_idx["train"]], preprocess=graph_preprocess_fns)
    valid_data = GraphDataset([dataset[i] for i in split_idx["valid"]], preprocess=graph_preprocess_fns)
    test_data = GraphDataset([dataset[i] for i in split_idx["test"]], preprocess=graph_preprocess_fns)

    collate_fn = partial(mol_graph_collate, dataset_name=args.dataset)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    return train_loader, valid_loader, test_loader

