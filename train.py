from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import wandb
import time

from ogb.graphproppred import GraphPropPredDataset
from ogb.graphproppred import Evaluator


from utils import parse_args, hiv_graph_collate, count_parameters, LPE, GraphDataset
from models.perceiver_graph_models import HIVModel, HIVModelNodeOnly


evaluator = Evaluator(name="ogbg-molhiv")


def run_epoch(model, iterator, optimizer, clip, criterion, device, mode="train"):
    if mode == "train":
        model.train()
        optimizer.zero_grad()
    elif mode == "val" or mode == "test":
        model.eval()
        assert(optimizer is None)
        assert(clip is None)
    else:
        raise Exception("Invalid mode provided")

    epoch_loss = 0
    num_samples = 0
    accuracy = 0
    evaluator_dict = {'y_true': [], 'y_pred': []}
    for i, (batch_X, X_mask, batch_y) in enumerate(iterator):
        # forward pass
        output = model(batch_X, X_mask, device)
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
        evaluator_dict['y_true'] += batch_y.tolist()
        evaluator_dict['y_pred'] += (output[:, 1] - output[:, 0]).tolist()
        if i % 100 == 99:
            print(f"Finished batch {i + 1} in mode {mode}")

    evaluator_dict['y_true'] = torch.as_tensor(evaluator_dict['y_true']).unsqueeze(-1)
    evaluator_dict['y_pred'] = torch.as_tensor(evaluator_dict['y_pred']).unsqueeze(-1)
    return epoch_loss / num_samples, accuracy / num_samples, evaluator.eval(evaluator_dict)['rocauc']


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

    graph_preprocess_fns = [LPE(args.k_eigs)] if args.k_eigs > 0 else []
    train_data = GraphDataset([dataset[i] for i in split_idx["train"]], preprocess=graph_preprocess_fns)
    valid_data = GraphDataset([dataset[i] for i in split_idx["valid"]], preprocess=graph_preprocess_fns)
    test_data = GraphDataset([dataset[i] for i in split_idx["test"]], preprocess=graph_preprocess_fns)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=hiv_graph_collate)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True, collate_fn=hiv_graph_collate)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, collate_fn=hiv_graph_collate)


    with wandb.init(project="GraphPerceiver", entity="wzhang2022", config=args):
        wandb.run.name = args.run_name
        model = HIVModel(atom_emb_dim=args.atom_emb_dim, bond_emb_dim=args.bond_emb_dim, node_preprocess_dim=args.k_eigs,
                         p_depth=args.depth, p_latent_trsnfmr_depth=args.latent_transformer_depth,
                         p_num_latents=args.num_latents, p_latent_dim=args.latent_dim,
                         p_cross_heads=args.cross_heads, p_latent_heads=args.latent_heads,
                         p_cross_dim_head=args.cross_dim_head, p_latent_dim_head=args.latent_dim_head,
                         p_attn_dropout=args.attn_dropout, p_ff_dropout=args.ff_dropout,
                         p_weight_tie_layers=args.weight_tie_layers).to(device)
        
        print(f"Model has {count_parameters(model)} parameters")
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)
        criterion = nn.CrossEntropyLoss(reduction="mean", weight=torch.as_tensor([1232 / 32901, 1]).to(device)) # correct for class imbalance in HIV dataset
        # wandb.watch(model, criterion, log="all", log_freq=1000)
        best_valid_loss = float('inf')
        for epoch in range(args.n_epochs):

            start_time = time.time()

            train_loss, train_accuracy, train_roc = run_epoch(model, train_loader, optimizer, args.clip, criterion, device, "train")
            val_loss, val_accuracy, val_roc = run_epoch(model, valid_loader, None, None, criterion, device, "val")
            test_loss, test_accuracy, test_roc = run_epoch(model, test_loader, None, None, criterion, device, "test")
            scheduler.step()
            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if val_loss < best_valid_loss:
                best_valid_loss = val_loss
                torch.save(model.state_dict(), f"{args.save_file}.pt")

            wandb.log({"test_loss": test_loss, "val_loss": val_loss, "train_loss": train_loss,
                       "test_accuracy": test_accuracy, "val_accuracy": val_accuracy, "train_accuracy": train_accuracy,
                       "test_roc": test_roc, "val_roc": val_roc, "test_roc": test_roc})
            print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f}, Train ROC: {train_roc:.3f}')
            print(f'\t Val. Loss: {val_loss:.3f}, Val. ROC: {val_roc:.3f}, Test ROC: {test_roc:.3f}')
