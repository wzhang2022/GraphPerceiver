
import torch
import numpy as np
import wandb
import time

import random

from ogb.graphproppred import Evaluator

from utils import parse_args, count_parameters, make_model, make_criterion, make_optimizer, make_scheduler,\
    make_dataloaders


def run_epoch(model, iterator, optimizer, criterion, device, evaluator, mode, args):
    if mode == "train":
        model.train()
        try:
            model.frozen_layers.eval()  # during transfer learning, turn on evaluation mode for dropout layers
        except AttributeError:
            pass
        optimizer.zero_grad()
    elif mode == "val" or mode == "test":
        model.eval()
        assert(optimizer is None)
        assert(args is None)
    else:
        raise Exception("Invalid mode provided")

    epoch_loss = 0
    num_samples = 0
    accuracy = 0
    evaluator_dict = {'y_true': [], 'y_pred': []}
    for i, (batch_X, X_mask, batch_y) in enumerate(iterator):
        if args.num_flag_steps > 0:
            # forward pass for FLAG training
            node_features, node_preprocess_feat, edge_index, edge_features = batch_X
            pert = torch.FloatTensor(node_features.shape[0], node_features.shape[1], args.atom_emb_dim + args.k_eigs)
            pert = pert.to(device)
            pert.uniform_(-args.flag_step_size, args.flag_step_size)
            pert.requires_grad = True
            output = model(batch_X, X_mask, device, node_pert=pert)
            loss = criterion(output, batch_y.to(device)) / args.num_flag_steps
        else:
            # forward pass for normal training
            output = model(batch_X, X_mask, device)
            loss = criterion(output, batch_y.to(device))

        # backward pass(es)
        if mode == "train":
            for _ in range(args.num_flag_steps - 1):
                loss.backward()
                pert_data = pert.detach() + args.flag_step_size * torch.sign(pert.grad.detach())
                pert.data = pert_data.data
                pert.grad[:] = 0
                output = model(batch_X, X_mask, device, node_pert=pert)
                loss = criterion(output, batch_y.to(device)) / args.num_flag_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
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

    evaluator = Evaluator(name=f"ogbg-{args.dataset}")
    train_loader, valid_loader, test_loader = make_dataloaders(args)

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

            train_loss, train_accuracy, train_eval_metric = run_epoch(model, train_loader, optimizer, criterion,
                                                              device, evaluator, "train", args)
            val_loss, val_accuracy, val_eval_metric = run_epoch(model, valid_loader, None, criterion,
                                                        device, evaluator, "val", None)
            test_loss, test_accuracy, test_eval_metric = run_epoch(model, test_loader, None, criterion,
                                                           device, evaluator, "test", None)
            
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
