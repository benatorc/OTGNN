import argparse
import os

from utils import write_args
import pdb


DATA_TASK = {
    'sol_test': 'rmse',
    'sol': 'rmse',
    'lipo': 'rmse',
    'bace': 'auc',
    'bbbp': 'auc',
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cuda', action='store_true', default=False,
                        help='Use gpu')

    # Data/Output Directories Params
    parser.add_argument('-data', required=True, help='Task, see above')
    parser.add_argument('-output_dir', default='', help='Output directory')
    parser.add_argument('-log_tb', action='store_true', default=False,
                        help='Use tensorboard to log')
    parser.add_argument('-write_every', type=int, default=20,
                        help='Write val results every this many epochs')

    # Pre-trained Model Params
    parser.add_argument('-pretrain_gcn', type=str, default=None,
                        help='path to pretrained gcn to use in another model')
    parser.add_argument('-pretrain_model', type=str, default=None,
                        help='path to pretrained model to load')

    # General Model Params
    parser.add_argument('-n_splits', type=int, default=1,
                        help='Number of data splits to train on')
    parser.add_argument('-n_epochs', type=int, default=100,
                        help='Number of epochs to train on')
    parser.add_argument('-lr', type=float, default=1e-3,
                        help='Static learning rate of the optimizer')
    parser.add_argument('-separate_lr', action='store_true', default=False,
                        help='Whether to use different lr for pc')
    parser.add_argument('-lr_pc', type=float, default=1e-2,
                        help='The learning rate for point clouds')
    parser.add_argument('-pc_xavier_std', type=float, default=0.1)
    parser.add_argument('-batch_size', type=int, default=48,
                        help='Number of examples in each batch')
    parser.add_argument('-max_grad_norm', type=float, default=10,
                        help='Clip gradients with higher norm')
    parser.add_argument('-patience', type=int, default=-1,
                        help='Stop training if not improved for this many')

    # GCN Params
    parser.add_argument('-n_layers', type=int, default=5,
                        help='Number of layers in model')
    parser.add_argument('-n_hidden', type=int, default=128,
                        help='Size of hidden dimension for model')
    parser.add_argument('-n_ffn_hidden', type=int, default=100)
    parser.add_argument('-dropout_gcn', type=float, default=0.,
                        help='Amount of dropout for the model')
    parser.add_argument('-dropout_ffn', type=float, default=0.,
                        help='Dropout for final ffn layer')
    parser.add_argument('-agg_func', type=str, choices=['sum', 'mean'],
                        default='sum', help='aggregator function for atoms')
    parser.add_argument('-batch_norm', action='store_true', default=False,
                        help='Whether or not to normalize atom embeds')

    # Prototype Params
    parser.add_argument('-init_method', default='none',
                        choices=['none', 'various', 'data'])
    parser.add_argument('-distance_metric', type=str, default='wasserstein',
                        choices=['l2', 'wasserstein', 'dot'])
    parser.add_argument('-n_pc', type=int, default=2,
                        help='Number of point clouds')
    parser.add_argument('-pc_size', type=int, default=20,
                        help='Number of points in each point cloud')
    parser.add_argument('-pc_hidden', type=int, default=-1,
                        help='Hidden dim for point clouds, different from GCN hidden dim')
    parser.add_argument('-pc_free_epoch', type=int, default=0,
                        help='If intialized with data, when to free pc')
    parser.add_argument('-ffn_activation', type=str, choices=['ReLU', 'LeakyReLU'],
                        default='LeakyReLU')
    parser.add_argument('-mult_num_atoms', action='store_true', default=True,
                        help='Whether to multiply the dist by number of atoms')

    # OT Params
    parser.add_argument('-opt_method', type=str, default='sinkhorn_stabilized',
                        choices=['sinkhorn', 'sinkhorn_stabilized', 'emd',
                                 'greenkhorn', 'sinkhorn_epsilon_scaling'])
    parser.add_argument('-cost_distance', type=str, choices=['l2', 'dot'],
                        default='l2', help='Distance computed for cost matrix')
    parser.add_argument('-sinkhorn_entropy', type=float, default=1e-1,
                        help='Entropy regularization term for sinkhorn')
    parser.add_argument('-sinkhorn_max_it', type=int, default=1000,
                        help='Max num it for sinkhorn')
    parser.add_argument('-unbalanced', action='store_true', default=False)
    parser.add_argument('-nce_coef', type=float, default=0.)

    # Plot Params
    parser.add_argument('-plot_pc', action='store_true', default=False,
                        help='Whether to plot the point clouds')
    parser.add_argument('-plot_num_ex', type=int, default=5,
                        help='Number of molecule examples to plot')
    parser.add_argument('-plot_freq', type=int, default=10,
                        help='Frequency of plotting')
    parser.add_argument('-plot_max', type=int, default=1000,
                        help='Maximum number of plots to make')

    args = parser.parse_args()
    args.device = 'cuda:0' if args.cuda else 'cpu'

    # Add path to data dir
    assert args.data in DATA_TASK
    # task specifies the kind of data
    args.task = DATA_TASK[args.data]

    # get the number of labels from the dataset, split by commas
    args.n_labels = len(args.task.split(','))

    if args.n_labels == 1:
        # val_stat is the stat to select the best model
        args.val_stat = args.task
    else:
        # if multiple labels, use a "multi-objective," some average of individual objectives
        args.val_stat = 'multi_obj'
        args.label_task_list = args.task.split(',')

    args.data_dir = 'data/%s' % args.data

    # hacky way to create the output directory initally
    if '/' in args.output_dir:
        base_output_dir = args.output_dir.split('/')[0]
        if not os.path.exists(base_output_dir):
            os.makedirs(base_output_dir)

    if args.output_dir != '' and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.output_dir != '':
        write_args(args)
    return args
