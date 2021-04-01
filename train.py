from ogb.graphproppred import GraphPropPredDataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn as nn
import numpy as np
import wandb
import random
import time
import math


from utils import parse_args
from models.perceiver import Perceiver
from models.integer_vector_embedding import IntegerVectorEmbedding



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



PAD_IDX = 150

# set random seeds
SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def graph_collate(batch):
    node_features = pad_sequence([torch.as_tensor(item[0]['node_feat']) for item in batch], batch_first=True, padding_value=PAD_IDX)
    node_mask = (node_features != PAD_IDX)[:, :, 0]  # (batch_size, max_number_nodes)
    edge_index = pad_sequence([torch.as_tensor(item[0]['edge_index'].transpose()) for item in batch], batch_first=True, padding_value=PAD_IDX)
    edge_mask = (edge_index != PAD_IDX)[:, :, 0]     # (batch_size, max_number_edges)
    edge_features = pad_sequence([torch.as_tensor(item[0]['edge_feat']) for item in batch], batch_first=True, padding_value=PAD_IDX)
    labels = torch.Tensor([item[1][0] for item in batch]).long()
    return (node_features, edge_index, edge_features), (node_mask, edge_mask), labels


def transform_graph_to_input(batch_X, device):
    node_features, edge_index, edge_features = batch_X
    return node_features.to(device)


def train(model, iterator, optimizer, clip, criterion, device):
    model.train()
    epoch_loss = 0
    for i, (batch_X, X_mask, batch_y) in enumerate(iterator):
        optimizer.zero_grad()
        output = model(transform_graph_to_input(batch_X, device))
        loss = criterion(output, batch_y.to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item() * len(batch_y)
        if i % 50 == 49:
            print(f"Finished batch {i + 1} training")

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion, device):
    model.eval()
    epoch_loss = 0
    accuracy = 0
    with torch.no_grad():
        for i, (batch_X, X_mask, batch_y) in enumerate(iterator):
            batch_y = batch_y.to(device)
            output = model(transform_graph_to_input(batch_X, device))
            accuracy += sum(batch_y == torch.argmax(output, dim=1))
            loss = criterion(output, batch_y)
            epoch_loss += loss.item() * len(batch_y)
            if i % 50 == 49:
                print(f"Finished batch {i + 1} validation")
    return epoch_loss / len(iterator), accuracy / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs



if __name__ == "__main__":
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = GraphPropPredDataset(name="ogbg-molhiv", root='dataset/')
    split_idx = dataset.get_idx_split()
    train_loader = DataLoader([dataset[i] for i in split_idx["train"]], batch_size=args.batch_size, shuffle=True, collate_fn=graph_collate)
    valid_loader = DataLoader([dataset[i] for i in split_idx["valid"]], batch_size=args.batch_size, shuffle=False, collate_fn=graph_collate)
    test_loader = DataLoader([dataset[i] for i in split_idx["test"]], batch_size=args.batch_size, shuffle=False, collate_fn=graph_collate)

    with wandb.init(project="GraphPerceiver", config=args):
        wandb.run.name = args.run_name
        model = nn.Sequential(IntegerVectorEmbedding(input_dim=9, output_dim=256, num_embedding=PAD_IDX + 1, pad_idx=PAD_IDX),
                              Perceiver(input_dim=256, depth=args.depth, num_classes=2)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        criterion = nn.CrossEntropyLoss(reduction="mean")
        wandb.watch(model, criterion, log="all", log_freq=1000)
        best_valid_loss = float('inf')

        for epoch in range(args.n_epochs):

            start_time = time.time()

            train_loss = train(model, train_loader, optimizer, args.clip, criterion, device)
            valid_loss, valid_accuracy = evaluate(model, valid_loader, criterion, device)

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), f"{args.save_file}.pt")

            wandb.log({"validation_loss": valid_loss, "train_loss": train_loss, "accuracy": valid_accuracy})
            print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
