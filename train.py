from ogb.graphproppred import GraphPropPredDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import wandb
import time
import math


from utils import parse_args, hiv_graph_collate, count_parameters
from models.perceiver_graph_models import HIVModel


def transform_graph_to_input(batch_X, device):
    node_features, edge_index, edge_features = batch_X
    return node_features.to(device)


def run_epoch(model, iterator, optimizer, clip, criterion, device, mode="train"):
    if mode == "train":
        model.train()
        optimizer.zero_grad()
    elif mode == "eval":
        model.eval()
        assert(optimizer is None)
        assert(clip is None)
    else:
        raise Exception("Invalid mode provided")

    epoch_loss = 0
    num_samples = 0
    accuracy = 0
    for i, (batch_X, X_mask, batch_y) in enumerate(iterator):
        # forward pass
        output = model(transform_graph_to_input(batch_X, device), X_mask[0].to(device))
        loss = criterion(output, batch_y.to(device))

        # backward pass
        if mode == "train":
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            optimizer.zero_grad()

        # record metrics
        accuracy += sum(batch_y.to(device) == torch.argmax(output, dim=1))
        epoch_loss += loss.item() * len(batch_y)
        num_samples += len(batch_y)
        if i % 100 == 99:
            print(f"Finished batch {i + 1} in mode {mode}")

    return epoch_loss / num_samples, accuracy / num_samples


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
    train_loader = DataLoader([dataset[i] for i in split_idx["train"]], batch_size=args.batch_size, shuffle=True, collate_fn=hiv_graph_collate)
    valid_loader = DataLoader([dataset[i] for i in split_idx["valid"]], batch_size=args.batch_size, shuffle=False, collate_fn=hiv_graph_collate)
    test_loader = DataLoader([dataset[i] for i in split_idx["test"]], batch_size=args.batch_size, shuffle=False, collate_fn=hiv_graph_collate)

    with wandb.init(project="GraphPerceiver", config=args):
        wandb.run.name = args.run_name
        model = HIVModel(atom_emb_dim=64, perceiver_depth=3).to(device)
        print(f"Model has {count_parameters(model)} parameters")
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
        criterion = nn.CrossEntropyLoss(reduction="mean")
        wandb.watch(model, criterion, log="all", log_freq=1000)
        best_valid_loss = float('inf')

        for epoch in range(args.n_epochs):

            start_time = time.time()

            train_loss, train_accuracy = run_epoch(model, train_loader, optimizer, args.clip, criterion, device, "train")
            valid_loss, valid_accuracy = run_epoch(model, valid_loader, None, None, criterion, device, "eval")

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), f"{args.save_file}.pt")

            wandb.log({"validation_loss": valid_loss, "train_loss": train_loss,
                       "valid_accuracy": valid_accuracy, "train_accuracy": train_accuracy})
            print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f}, Train Accuracy: {train_accuracy:.3f}')
            print(f'\t Val. Loss: {valid_loss:.3f}, Val. Accuracy: {valid_accuracy:.3f}')
