from collections import defaultdict

import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from datasets import get_loader
from graph import MolGraph
from matplotlib import gridspec

import pdb

def read_data(data_path, parse_func=None):
    data = []
    with open(data_path, 'r+') as data_file:
        for line in data_file.readlines():
            if parse_func is None:
                smiles, label = line.strip().split(',')
                data.append((smiles, label))
            else:
                data.append(parse_func(line.strip()))
    return data


def get_data_examples(args, data_type, n_examples, shuffle_data=False):
    """Returns examples from data"""
    data_loader = get_loader(
        args.data_dir, data_type=data_type, batch_size=n_examples,
        shuffle=shuffle_data, split=0, n_labels=args.n_labels)
    examples = data_loader.__iter__().next()
    return examples

class PlotTracker(object):
    # Class to keep track of point cloud embeddings through training
    def __init__(self, plot_freq, plot_max, save_dir, title, pc_names,
                 text_names):
        """Initializes the plot tracker for first two dim of point clouds

        Args:
            plot_freq: How often to save data. If data is added every batch,
                data is saved every #plot_freq batches
            plot_max: Number of max data points to save
            save_dir: Save directory, should be the same as output directory
            title: Title of the plot
            pc_names: The names of each point cloud to be stored
            text_names: The names of the text data to store
        """
        self.plot_freq = plot_freq
        self.plot_max = plot_max

        self.save_dir = save_dir
        self.title = title

        self.idx = 0  # Internal counter
        self.n_plots = 0 # Num plots added

        self.pc_data = []  # Keep track of pc data
        self.text_data = {}  # Keep track of the text data

        # Hacky way to cycle around different colors for each point cloud
        # Note that if the number of point clouds exceed the preset colors,
        # will cycle back to the first color
        self.color_wheel_1 = [
            'ro', 'bo', 'go', 'co', 'mo', 'r+', 'b+', 'g+', 'c+', 'm+', \
            'r^', 'b^', 'g^', 'c^', 'm^', 'rs', 'bs', 'gs', 'cs', 'ms']
        self.color_wheel_2 = ['kp', 'yp', 'kd', 'yd', 'kx', 'yx']

        # Keep track of the dimensions of the plot, constantly updated
        self.plot_dims = [0, 0, 0, 0]  # x_min, x_max, y_min, y_max

        # Store the names of the text data
        self.text_names = text_names

        # Store the names of the point clouds and keep track of number:
        self.pc_names = pc_names
        self.n_pc = len(pc_names)

    def update_dims(self, coords):
        """Updates the coordinates of plot."""
        if coords[0] < self.plot_dims[0]:
            self.plot_dims[0] = coords[0]

        if coords[1] > self.plot_dims[1]:
            self.plot_dims[1] = coords[1]

        if coords[2] < self.plot_dims[2]:
            self.plot_dims[2] = coords[2]

        if coords[3] > self.plot_dims[3]:
            self.plot_dims[3] = coords[3]

    def add_pc_list(self, pc_list, text_dict=None):
        """Add a list of point clouds data."""
        # Number of point clouds added should be consistent
        assert len(pc_list) == self.n_pc

        if self.n_plots >= self.plot_max:
            # Do not add if the number of plots exceed max
            return

        if self.idx % self.plot_freq != 0:
            # Only save data every #plot_freq
            self.idx += 1
            return

        self.idx += 1
        self.n_plots += 1

        def parse_pc_list(pc_list):
            new_data = []
            # Iterate through the point clouds
            for pc_idx, pc_X in enumerate(pc_list):
                # Only take the first two dimensions
                pc_X = pc_list[pc_idx].clone().cpu().numpy()[:, :2]
                x_data, y_data = pc_X[:, 0], pc_X[:, 1]

                min_x, max_x = min(x_data), max(x_data)
                min_y, max_y = min(y_data), max(y_data)
                self.update_dims([min_x, max_x, min_y, max_y])

                new_data.append((x_data, y_data))
            return new_data

        new_pc_data = parse_pc_list(pc_list)
        self.pc_data.append(new_pc_data)

        for name, val in text_dict.items():
            if name in self.text_data:
                self.text_data[name].append(val)
            else:
                self.text_data[name] = [val]


    def get_args_string(self, args):
        if args is None:
            return ''

        PROPS_DICT = {
            'GCN': ['n_hidden', 'n_layers', 'dropout', 'agg_func'],
            'OPTIM': ['lr', 'separate_lr', 'lr_pc', 'batch_size'],
            'PC': ['distance_metric', 'e_dist', 'n_pc', 'pc_size', 'pc_hidden',
                   'dropout_pc', 'ffn_activation', 'init_method'],
            'OT': ['opt_method', 'sinkhorn_entropy', 'sinkhorn_max_it',
                   'cost_distance']
        }

        args_string = ''
        for category_name in sorted(PROPS_DICT):
            name_list = PROPS_DICT[category_name]
            cur_string = '%s:\n' % category_name
            for name in name_list:
                if name in args.__dict__:
                    val = args.__dict__[name]
                    cur_string += '%s: %s\n' % (name, str(val))
            args_string += cur_string + '\n'
        return args_string

    def plot_and_save(self, args=None):
        """Construct and save the plot."""
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

        fig = plt.figure(figsize=(12, 8))
        gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
        ax = plt.subplot(
            gs[0], xlim=(self.plot_dims[0] - 0.5, self.plot_dims[1] + 0.5),
            ylim=(self.plot_dims[2] - 0.5, self.plot_dims[3] + 0.5))
        ax_text = plt.subplot(gs[1])
        ax_text.axis('off')
        plt.title(self.title, fontdict={'fontsize': 20, 'fontweight': 5})  # Set Title

        text_list = []  # Create graph text list
        for idx in range(len(self.text_data)):
            text_ele = ax.text(0.02, 1 - (0.05 * (idx + 1)), '', transform=ax.transAxes)
            text_list.append(text_ele)

        plot_list = []  # Create graph plot list
        for idx in range(self.n_pc):
            plot_name = self.pc_names[idx]
            # Hacky way to easily choose from another color wheel
            color_wheel = self.color_wheel_2 if 'ex' in plot_name.lower() else self.color_wheel_1
            plot, = ax.plot([], [], color_wheel[idx % len(color_wheel)])
            plot_list.append(plot)

        # Create graph legend
        plt.legend(plot_list, self.pc_names, bbox_to_anchor=(-0.2, 1),
                   loc='upper left', borderaxespad=0.)

        ax_text.text(1.30, 0.60, s=self.get_args_string(args),
                     transform=ax.transAxes, fontsize=8.5,
                     verticalalignment='top')

        def animate(i):
            data = self.pc_data[i]
            for j in range(self.n_pc):
                x_data, y_data = data[j][0], data[j][1]
                plot_list[j].set_data(x_data, y_data)

            for j, name in enumerate(self.text_names):
                text_list[j].set_text('%s: %s' % (name, self.text_data[name][i]))
            return plot_list

        plot_ani = animation.FuncAnimation(
            fig, animate, frames=len(self.pc_data), blit=True)
        save_loc = '%s/plot.mp4' % self.save_dir
        plot_ani.save(save_loc, writer=writer)

        print('Plot saved to: %s' % save_loc)

class StatsTracker(object):
    # Tracks stats across a single epoch
    def __init__(self):
        self.stats_sum = defaultdict(float)
        self.stats_norm = defaultdict(float)

    def add_stat(self, stat_name, val, norm=1):
        self.stats_sum[stat_name] += val
        self.stats_norm[stat_name] += norm

    def get_stats(self):
        stats = {
            name: self.stats_sum[name] / self.stats_norm[name]
            for name in self.stats_sum
        }
        return stats

    def print_stats(self, pre=''):
        stats = self.get_stats()
        stats_string = ''
        for name, val in stats.items():
            stats_string += '%s: %.4f ' % (name, val)
        stats_string = stats_string[:-1]
        print(pre + ' ' + stats_string)


class StatsManager(object):
    # Tracks stats across the model run
    def __init__(self):
        self.names = []
        self.stats = {}

    def add_stats(self, stats):
        for n, v in stats.items():
            if n not in self.names:
                self.names.append(n)
                self.stats[n] = []
            self.stats[n].append(v)

    def get_best_epoch_for_stat(self, stat_name, lower_better=True):
        stat_list = self.stats[stat_name]

        if lower_better:
            best_idx = stat_list.index(min(stat_list))
        else:
            best_idx = stat_list.index(max(stat_list))

        epoch = int(self.stats['epoch'][best_idx])
        return epoch
