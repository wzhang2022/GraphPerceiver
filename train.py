from torch.utils.data import DataLoader
import torch
import numpy as np
import wandb
import time
from functools import partial
import random

from ogb.graphproppred import GraphPropPredDataset
from ogb.graphproppred import Evaluator

from utils import parse_args, mol_graph_collate, count_parameters, LPE, GraphDataset, make_model, make_criterion, \
    make_optimizer, make_scheduler


def run_epoch(model, iterator, optimizer, clip, criterion, device, evaluator, mode="train"):
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
        epoch_loss += loss.item() * len(batch_y)
        num_samples += len(batch_y)
        if evaluator.name == "ogbg-molhiv":
            accuracy += sum(batch_y.to(device) == torch.argmax(output, dim=1))
            evaluator_dict['y_true'] += batch_y.tolist()
            evaluator_dict['y_pred'] += (output[:, 1] - output[:, 0]).tolist()
        elif evaluator.name == "ogbg-molpcba":
            # TODO: implement accuracy accumulation for molpcba dataset
            accuracy = -1  * num_samples                         # while not implemented return negative number so we know      
            evaluator_dict['y_true'] += batch_y.tolist()
            evaluator_dict['y_pred'] += output.tolist()
        # log progress for viewing during training/testing
        if i % 100 == 99:
            print(f"Finished batch {i + 1} in mode {mode}")

    if evaluator.name == "ogbg-molhiv":
        evaluator_dict['y_true'] = torch.as_tensor(evaluator_dict['y_true']).unsqueeze(-1)
        evaluator_dict['y_pred'] = torch.as_tensor(evaluator_dict['y_pred']).unsqueeze(-1)
    elif evaluator.name == "ogbg-molpcba":
        evaluator_dict['y_true'] = torch.as_tensor(evaluator_dict['y_true'])
        evaluator_dict['y_pred'] = torch.as_tensor(evaluator_dict['y_pred'])
    return epoch_loss / num_samples, accuracy / num_samples, evaluator.eval(evaluator_dict)[evaluator.eval_metric]


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


if __name__ == "__main__":
    args = parse_args()

    # set random seeds
    SEED = args.seed

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True


    assert args.dataset in ['molhiv', 'molpcba']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # preprocessing: start
    pp_start_time = time.time()

    dataset = GraphPropPredDataset(name=f"ogbg-{args.dataset}", root='dataset/')
    evaluator = Evaluator(name=f"ogbg-{args.dataset}")
    if args.shuffle_split:
        indices = np.random.permutation(len(dataset))
        idx1, idx2 = int(len(dataset) * 0.8), int(len(dataset) * 0.9)
        train, valid, test = indices[:idx1], indices[idx1:idx2], indices[idx2:]
        split_idx = {"train": train, "valid": valid, "test": test}
    else:
        split_idx = dataset.get_idx_split()

    graph_preprocess_fns = [LPE(args.k_eigs)] if args.k_eigs > 0 else []
    train_data = GraphDataset([dataset[i] for i in split_idx["train"]], preprocess=graph_preprocess_fns)
    valid_data = GraphDataset([dataset[i] for i in split_idx["valid"]], preprocess=graph_preprocess_fns)
    test_data = GraphDataset([dataset[i] for i in split_idx["test"]], preprocess=graph_preprocess_fns)

    collate_fn = partial(mol_graph_collate, dataset_name=args.dataset)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    
    pp_end_time = time.time()
    pp_mins, pp_secs = epoch_time(pp_start_time, pp_end_time)
    print(f'Preprocessing time: {pp_mins}m {pp_secs}s')
    
    with wandb.init(project="GraphPerceiver", entity="wzhang2022", config=args):
        wandb.run.name = args.run_name
        model = make_model(args).to(device)
        
        print(f"Model has {count_parameters(model)} parameters")
        print()

        optimizer = make_optimizer(args, model)
        scheduler = make_scheduler(args, optimizer)
        criterion = make_criterion(args).to(device)
        # wandb.watch(model, criterion, log="all", log_freq=1000)
        best_valid_loss = float('inf')
        
        # begin training
        for epoch in range(args.n_epochs):

            start_time = time.time()

            train_loss, train_accuracy, train_eval_metric = run_epoch(model, train_loader, optimizer, args.clip, criterion,
                                                              device, evaluator, "train")
            val_loss, val_accuracy, val_eval_metric = run_epoch(model, valid_loader, None, None, criterion,
                                                        device, evaluator, "val")
            test_loss, test_accuracy, test_eval_metric = run_epoch(model, test_loader, None, None, criterion,
                                                           device, evaluator, "test")
            
            scheduler.step()
            
            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if val_loss < best_valid_loss:
                best_valid_loss = val_loss
                torch.save(model.state_dict(), f"{args.save_file}.pt")

            metric = evaluator.eval_metric
            wandb.log({"test_loss": test_loss, "val_loss": val_loss, "train_loss": train_loss,
                       "test_accuracy": test_accuracy, "val_accuracy": val_accuracy, "train_accuracy": train_accuracy,
                       f"test_{metric}": test_eval_metric, f"val_{metric}": val_eval_metric,
                       f"train_{metric}": train_eval_metric})
            
            # print metrics after every epoch
            print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\t Train Loss: {train_loss:.3f}\tTrain {metric}: {train_eval_metric:.3f}')
            print(f'\t Val. Loss: {val_loss:.3f} \tVal {metric}: {val_eval_metric:.3f}')
            print(f'\t Test Loss: {test_loss:.3f} \tTest {metric} {test_eval_metric:.3f}')
