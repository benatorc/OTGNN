import json
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from arguments import get_args
from utils import StatsTracker, PlotTracker, get_data_examples

from train_base import train_model, test_model

from graph import MolGraph
from datasets import get_loader
from models import ProtoNet
import pdb


def init_plot_tracker(args):
    pc_names = []
    for i in range(args.n_pc):
        # Add the names of the point clouds
        if args.init_method == 'data':
            pc_names.append('pc %d, (%.2f)' % (i, args.pc_data[1][i]))
        else:
            pc_names.append('pc %d' % i)

    if args.plot_num_ex > 0:
        args.ex_data = get_data_examples(
            args, 'test', args.plot_num_ex, shuffle_data=True)

    for i in range(args.plot_num_ex):
        pc_names.append('ex %d (%s)' % (i, str(args.ex_data[1][i])))

    # Text data to store are the epoch and the val_stat, either rmse or auc
    text_names = ['epoch', 'train %s' % args.val_stat, 'val %s' % args.val_stat]
    args.plot_tracker = PlotTracker(
        plot_freq=args.plot_freq, plot_max=args.plot_max,
        save_dir=args.output_dir, title=args.data, text_names=text_names,
        pc_names=pc_names)

def main(args=None, quiet=False, splits=None, abs_output_dir=False):
    if args is None:
        args = get_args()

    if args.pretrain_model is not None:
        test_model(model_class=ProtoNet, run_func=run_func, args=args, quiet=quiet)
        exit()

    # If should initialize the point clouds as data points, get random mols
    # from the validation set
    if args.init_method == 'data':
        train_data_loader = get_loader(
            args.data_dir, data_type='val', batch_size=args.n_pc,
            shuffle=True, split=0, n_labels=args.n_labels)
        pc_data = train_data_loader.__iter__().next()
        args.pc_data = pc_data

    if args.plot_pc:
        init_plot_tracker(args)

    train_results = train_model(
        model_class=ProtoNet, run_func=run_func, args=args, quiet=quiet,
        splits=splits, abs_output_dir=abs_output_dir)

    if args.plot_pc:
        args.plot_tracker.plot_and_save(args)

    return train_results

def run_func(model, optim, data_loader, data_type, args, write_path=None,
             debug=False, quiet=False):
    is_train = data_type == 'train'
    if is_train:
        model.train()
    else:
        model.eval()

    # Keeps track of epoch stats by aggregating batch stats
    stats_tracker = StatsTracker()
    all_preds, all_labels = [], []
    if args.n_labels > 1:
        for _ in range(args.n_labels):
            all_preds.append([])
            all_labels.append([])
    write_output = []

    # Add the pc ot distance of the first two point clouds
    for _, batch_data in enumerate(tqdm(data_loader, disable=quiet)):
        if is_train:
            optim.zero_grad()

        if args.n_labels == 1:
            smiles_list, labels_list = batch_data
        else:
            smiles_list, labels_list, mask = batch_data
        n_data = len(smiles_list)

        labels = torch.tensor(labels_list, device=args.device).float()
        if args.n_labels == 1:
            labels = labels.unsqueeze(1)
        if args.n_labels > 1:
            mask = torch.tensor(mask, device=args.device).float()

        if data_type == 'train' and args.plot_pc:
            with torch.no_grad():
                plot_tracker = args.plot_tracker

                if model.pc_list is None:  # Hack for the first batch
                    model.pc_list = model.get_graph_pc_embeddings(MolGraph(args.pc_data[0]))

                pc_data = []
                for i in range(args.n_pc):
                    pc_data.append(model.pc_list[i].detach())

                if args.plot_num_ex > 0:
                    ex_smiles = args.ex_data[0]
                    ex_pc_list = model.get_graph_pc_embeddings(MolGraph(ex_smiles))
                pc_data += ex_pc_list

                text_data = {
                    'epoch': args.epoch,
                    'train %s' % args.val_stat: args.latest_train_stat,
                    'val %s' % args.val_stat: args.latest_val_stat}

                plot_tracker.add_pc_list(pc_data, text_data)

        # mol_graph is the abstraction of the model input
        mol_graph = MolGraph(smiles_list)
        output_dict = model(mol_graph, debug=debug)

        preds = output_dict['preds']
        if args.val_stat == 'rmse':
            loss = torch.mean((labels - preds) ** 2)
            stats_tracker.add_stat('mse', loss.item() * n_data, n_data)
            all_preds.append(preds.detach().cpu().numpy())
        elif args.val_stat in ['auc', 'acc']:
            pred_probs = nn.Sigmoid()(preds)
            loss = nn.BCELoss()(input=pred_probs, target=labels)
            stats_tracker.add_stat('ce', loss.item() * n_data, n_data)

            # Aggregate predictions to compute Acc/AUC
            all_preds.append(pred_probs.detach().cpu().numpy())
        elif args.val_stat == 'mae':
            loss = torch.mean(torch.abs(labels - preds))
            stats_tracker.add_stat('mae', loss.item() * n_data, n_data)
            all_preds.append(preds.detach().cpu().numpy())
        elif args.val_stat == 'multi_obj':
            # iterate through each label and compute the loss and aggregate
            total_loss = 0
            for i in range(args.n_labels):
                cur_preds = preds[:, i].unsqueeze(1)
                cur_labels = labels[:, i].unsqueeze(1)
                cur_mask = mask[:, i].unsqueeze(1)

                n_sum = torch.sum(cur_mask).item()
                if n_sum == 0:
                    continue

                all_preds[i].append(cur_preds[cur_mask.byte()].detach().cpu().numpy())
                all_labels[i].append(cur_labels[cur_mask.byte()].detach().cpu().numpy())
                task_stat = args.label_task_list[i]
                if task_stat == 'rmse':
                    loss = (cur_preds - cur_labels) ** 2
                    loss = torch.sum(loss * cur_mask)
                    total_loss += loss
                if task_stat in ['auc', 'acc']:
                    loss = nn.BCEWithLogitsLoss(reduction='none')(input=cur_preds, target=cur_labels)
                    loss = torch.sum(loss * cur_mask)
                    total_loss += loss
                else:
                    assert False

            loss = total_loss / n_data
        else:
            assert False
        all_labels.append(np.array(labels_list))

        if args.nce_coef > 0.:
            nce_reg = output_dict['nce_reg']
            stats_tracker.add_stat('nce_reg', nce_reg.item() * n_data, n_data)
            loss = loss - args.nce_coef * nce_reg

        if write_path is not None and args.n_labels == 1:
            for smiles_idx, smiles in enumerate(smiles_list):
                label = labels[smiles_idx].item()
                pred = preds[smiles_idx].item()
                write_output.append({'smiles': smiles, 'label': label, 'pred': pred})

        # Add stats to keep track of
        stats_tracker.add_stat('loss', loss.item() * n_data, n_data)

        if is_train:
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optim.step()

    if args.n_labels == 1:
        all_preds = np.squeeze(np.concatenate(all_preds), axis=1)
        all_labels = np.concatenate(all_labels)

        preds_mean = np.mean(all_preds)
        preds_std = np.std(all_preds)
        stats_tracker.add_stat('prediction_mean', preds_mean, 1)
        stats_tracker.add_stat('prediction_std', preds_std, 1)

        if args.val_stat == 'rmse':
            mse = stats_tracker.get_stats()['mse']
            stats_tracker.add_stat('rmse', mse ** 0.5, 1)
        elif args.val_stat in ['auc', 'acc']:
            acc = np.mean((all_preds > 0.5) == all_labels)
            stats_tracker.add_stat('acc', -1 * acc, 1)

            auc = roc_auc_score(y_true=all_labels, y_score=all_preds)
            stats_tracker.add_stat('auc', -1 * auc, 1)
    else:
        all_stats = []
        for label_idx in range(args.n_labels):
            cur_preds = np.concatenate(all_preds[label_idx])
            cur_labels = np.concatenate(all_labels[label_idx])

            data_name = args.data.split('_')[label_idx]
            if args.label_task_list[i] == 'rmse':
                mse = np.mean((cur_preds - cur_labels) ** 2)
                stats_tracker.add_stat('%s_mse' % data_name, mse, 1)
                stats_tracker.add_stat('%s_rmse' % data_name, mse ** 0.5, 1)
            elif args.label_task_list[i] in ['auc', 'acc']:
                cur_pred_probs = 1 / (1 + np.exp(-cur_preds))
                acc = np.mean((cur_pred_probs > 0.5) == cur_labels)

                auc = roc_auc_score(y_true=cur_labels, y_score=cur_pred_probs)
                stats_tracker.add_stat('%s_acc' % data_name, acc, 1)
                stats_tracker.add_stat('%s_auc' % data_name, auc, 1)

                all_stats.append(-1 * auc)
            else:
                assert False
        stats_tracker.add_stat('multi_obj', np.mean(np.array(all_stats)), 1)

    if write_path is not None and args.n_labels == 1:
        with open(write_path, 'w+') as write_file:
            for write_ele in write_output:
                json.dump(write_ele, write_file)
                write_file.write('\n')

    return stats_tracker

def reshape_tensor_to_bond_list(list_of_tensors):
    """
    Input: list of tensors i each of shape (ni, ni, d)
    Output: list of bond lists each of shape [bond1,...,bond_m] with bond_j.shape=(d)
    """
    bond_list = []
    for i in range(len(list_of_tensors)):
        tensor = list_of_tensors[i]
        current_list = []
        n = tensor.shape[0]
        assert tensor.shape[1] == n
        for j in range(n):
            for k in range(n):
                current_list.append(tensor[j][k])
        bond_list.append(torch.stack(current_list))
    return bond_list

if __name__ == '__main__':
    main()
