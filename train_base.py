import os
import torch
import numpy as np
from tensorboardX import SummaryWriter

from datasets import get_loader
from utils import StatsManager
from utils import log_tensorboard
from utils import save_model, load_model, get_num_params, write_args

import pdb


def test_model(model_class, run_func, args, split_idx=0, quiet=False):
    output_dir = args.output_dir  # save the output_dir

    if not quiet:
        print('Model loaded from: %s' % args.pretrain_model)
    model, args = load_model(args.pretrain_model, model_class=model_class,
                             device=args.device)
    args.output_dir = output_dir

    test_data_loader = get_loader(
        args.data_dir, data_type='test', batch_size=args.batch_size,
        shuffle=False, split=split_idx, n_labels=args.n_labels)

    test_stats = run_func(
        model=model, optim=None, data_loader=test_data_loader, data_type='test',
        args=args, write_path='%s/test_output.jsonl' % args.output_dir, quiet=quiet)
    if not quiet:
        test_stats.print_stats('Test: ')


def train_model(model_class, run_func, args, quiet=False, splits=None, abs_output_dir=False):
    output_dir = args.output_dir

    val_stat = args.val_stat
    # Keeps track of certain stats for all the data splits
    all_stats = {
        'val_%s' % val_stat: [],
        'test_%s' % val_stat: [],
        'best_epoch': [],
        'train_last': [],
        'train_best': [],
        'nce': [],
    }

    # Iterate over splits
    splits_iter = splits if splits is not None else range(args.n_splits)
    # Iterates through each split of the data
    for split_idx in splits_iter:
        # print('Training split idx: %d' % split_idx)

        # Creates the output directory for the run of the current split
        if not abs_output_dir:
            args.output_dir = output_dir + '/run_%d' % split_idx
        args.model_dir = args.output_dir + '/models'
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        if not os.path.exists(args.model_dir):
            os.makedirs(args.model_dir)
        write_args(args)

        # Create model and optimizer
        model = model_class(args)
        model.to(args.device)

        if args.separate_lr:
            optim = model.get_model_optim()
        else:
            optim = torch.optim.Adam(model.parameters(), lr=args.lr)

        if split_idx == 0:
            # Print the number of parameters
            num_params = get_num_params(model)
            if not quiet:
                print('Initialized model with %d params' % num_params)

        # Load the train, val, test data
        dataset_loaders = {}
        for data_type in ['train', 'val', 'test']:
            dataset_loaders[data_type] = get_loader(
                args.data_dir, data_type=data_type, batch_size=args.batch_size,
                shuffle=data_type == 'train', split=split_idx,
                n_labels=args.n_labels)

        # Keeps track of stats across all the epochs
        train_m, val_m = StatsManager(), StatsManager()

        # Tensorboard logging, only for the first run split
        if args.log_tb and split_idx == 0:
            log_dir = output_dir + '/logs'
            tb_writer = SummaryWriter(log_dir, max_queue=1, flush_secs=60)
            log_tensorboard(tb_writer, {'params': num_params}, '', 0)
        else:
            args.log_tb = False

        # Training loop
        args.latest_train_stat = 0
        args.latest_val_stat = 0  # Keeps track of the latest relevant stat
        patience_idx = 0
        for epoch_idx in range(args.n_epochs):
            args.epoch = epoch_idx
            train_stats = run_func(
                model=model, optim=optim, data_loader=dataset_loaders['train'],
                data_type='train', args=args, write_path=None, quiet=quiet)
            should_write = epoch_idx % args.write_every == 0
            val_stats = run_func(
                model=model, optim=None, data_loader=dataset_loaders['val'],
                data_type='val', args=args,
                write_path='%s/val_output_%d.jsonl' % (args.output_dir, epoch_idx) if should_write else None, quiet=quiet)

            if not quiet:
                train_stats.print_stats('Train %d: ' % epoch_idx)
                val_stats.print_stats('Val   %d: ' % epoch_idx)

            if args.log_tb:
                log_tensorboard(tb_writer, train_stats.get_stats(), 'train', epoch_idx)
                log_tensorboard(tb_writer, val_stats.get_stats(), 'val', epoch_idx)

            train_stats.add_stat('epoch', epoch_idx)
            val_stats.add_stat('epoch', epoch_idx)

            train_m.add_stats(train_stats.get_stats())
            val_m.add_stats(val_stats.get_stats())

            if val_stats.get_stats()[val_stat] == min(val_m.stats[val_stat]):
                save_model(model, args, args.model_dir, epoch_idx, should_print=not quiet)
                patience_idx = 0
            else:
                patience_idx += 1
                if args.patience != -1 and patience_idx >= args.patience:
                    print('Validation error has not improved in %d, stopping at epoch: %d' % (args.patience, args.epoch))
                    break

            # Keep track of the latest epoch stats
            args.latest_train_stat = train_stats.get_stats()[val_stat]
            args.latest_val_stat = val_stats.get_stats()[val_stat]

        # Load and save the best model
        best_epoch = val_m.get_best_epoch_for_stat(args.val_stat)
        best_model_path = '%s/model_%d' % (args.model_dir, best_epoch)
        model, _ = load_model(
            best_model_path, model_class=model_class, device=args.device)
        if not quiet:
            print('Loading model from %s' % best_model_path)

        save_model(model, args, args.model_dir, 'best', should_print=not quiet)

        # Test model
        test_stats = run_func(
            model=model, optim=None, data_loader=dataset_loaders['test'],
            data_type='test', args=args,
            write_path='%s/test_output.jsonl' % args.output_dir, quiet=quiet)
        if not quiet:
            test_stats.print_stats('Test: ')

        if args.log_tb:
            log_tensorboard(tb_writer, test_stats.get_stats(), 'test', 0)
            tb_writer.close()

        # Write test output to a summary file
        with open('%s/summary.txt' % args.output_dir, 'w+') as summary_file:
            for k, v in test_stats.get_stats().items():
                summary_file.write('%s: %.3f\n' % (k, v))

        # Aggregate relevant stats
        all_stats['val_%s' % val_stat].append(min(val_m.stats[val_stat]))
        all_stats['test_%s' % val_stat].append(test_stats.get_stats()[val_stat])
        all_stats['best_epoch'].append(best_epoch)

        all_stats['train_last'].append(train_m.stats[val_stat][-1])
        all_stats['train_best'].append(train_m.stats[val_stat][best_epoch])

        if args.nce_coef > 0:
            all_stats['nce'].append(train_m.stats['nce_reg'][best_epoch])

    # Write the stats aggregated across all splits
    with open('%s/summary.txt' % (output_dir), 'w+') as summary_file:
        summary_file.write('Num epochs trained: %d\n' % args.epoch)
        for name, stats_arr in all_stats.items():
            if stats_arr == []:
                continue
            stats_arr = np.array(stats_arr)
            stats_mean = np.mean(stats_arr)
            stats_std = np.std(stats_arr)
            summary_file.write('%s: %s, mean: %.3f, std: %.3f\n' % (
                name, str(stats_arr), stats_mean, stats_std))

    all_val_stats = np.array(all_stats['val_%s' % val_stat])
    all_test_stats = np.array(all_stats['test_%s' % val_stat])

    val_mean, val_std = np.mean(all_val_stats), np.std(all_val_stats)
    test_mean, test_std = np.mean(all_test_stats), np.std(all_val_stats)

    train_last = np.mean(np.array(all_stats['train_last']))
    train_best = np.mean(np.array(all_stats['train_best']))

    if args.nce_coef > 0:
        nce_loss = np.mean(np.array(all_stats['nce']))
    else:
        nce_loss = 0

    # Return stats
    return (val_mean, val_std), (test_mean, test_std), (train_last, train_best), nce_loss
