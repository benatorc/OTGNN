import argparse
import math
import torch

import numpy as np
from scipy.stats import spearmanr, pearsonr

from utils import load_model

from models import GCN
from models import ProtoNet
from models import compute_ot

from graph import MolGraph
from tqdm import tqdm
from datasets import get_loader
import matplotlib.pyplot as plt

import pdb

BATCH_SIZE = 50

def agg_func(atom_h, scope, args):
    mol_h = []
    for (st, le) in scope:
        cur_atom_h = atom_h.narrow(0, st, le)

        if args.agg_func == 'sum':
            mol_h.append(cur_atom_h.sum(dim=0))
        elif args.agg_func == 'mean':
            mol_h.append(cur_atom_h.mean(dim=0))
        else:
            assert(False)
    mol_h = torch.stack(mol_h)
    return mol_h

def compute_pc_dist(model, mol_h, args):
    dist_emb = []

    n_atoms = mol_h.size()[0]
    H_1 = np.ones([n_atoms]) / n_atoms
    for pc_idx in range(args.n_pc):
        pc_X = model.pc_list[pc_idx]
        if args.distance_metric == 'wasserstein':
            ot_dist, _, _, _ = compute_ot(
                X_1=mol_h, X_2=pc_X, H_1=H_1, H_2=model.h_list[pc_idx],
                device=args.device, opt_method=args.opt_method,
                sinkhorn_entropy=args.sinkhorn_entropy,
                sinkhorn_max_it=args.sinkhorn_max_it,
                cost_distance=args.cost_distance,
            )
            dist_emb.append(ot_dist.item() * pc_X.size()[0] * n_atoms / 100)
        elif args.distance_metric == 'l2':
            l2_dist = torch.sum((mol_h - pc_X.squeeze(0)) ** 2)
            dist_emb.append(l2_dist.item())
    return np.array(dist_emb)

def compute_r(model_type, model, args, test_data_loader, should_plot=False):
    dist_list = []  # Stores the embedding distances
    label_diff_list = []  # Stores the label differences

    for _, batch_data in enumerate(tqdm(test_data_loader)):
        smiles_list, labels_list = batch_data

        n_data = len(smiles_list)

        mol_graph = MolGraph(smiles_list)
        output = model(mol_graph)

        if model_type == 'gcn':
            atom_h = output[0]
            mol_h = agg_func(atom_h, mol_graph.scope, args)
        elif model_type == 'proto':
            atom_h = output['atom_h']
            if args.distance_metric != 'wasserstein':
                mol_h = agg_func(atom_h, mol_graph.scope, args)
            else:
                mol_h = []
                for (st, le) in mol_graph.scope:
                    cur_atom_h = atom_h.narrow(0, st, le)
                    mol_h.append(cur_atom_h)

        for i in range(n_data):
            if i == n_data - 1:
                break
            for j in range(i+1, n_data, 1):
                mol_1, mol_2 = mol_h[i], mol_h[j]

                if (args.distance_metric == 'wasserstein') or (args.distance_metric == 'l2'):
                    embd_1 = compute_pc_dist(model, mol_1, args)
                    embd_2 = compute_pc_dist(model, mol_2, args)
                    dist = np.sum((embd_1 - embd_2) ** 2)
                else:
                    dist = torch.sum((mol_1 - mol_2) ** 2).item()
                dist_list.append(dist)

                label_1, label_2 = labels_list[i], labels_list[j]
                label_diff = abs(label_1 - label_2)
                label_diff_list.append(label_diff)

    dist_arr = np.array(dist_list)
    label_diff_arr = np.array(label_diff_list)

    print('Total pairs: %d' % len(dist_list))

    s_rho, _ = spearmanr(dist_arr, label_diff_arr)
    p_rho, _ = pearsonr(dist_arr, label_diff_arr)

    print('Spearman correlation: %.3f' % s_rho)
    print('Pearson correlation: %.3f' % p_rho)

    if should_plot:
        plt.plot(dist_arr, label_diff_arr, 'bo')
        if model_type == 'gcn':
            model_name = 'gcn'
        else:
            if args.distance_metric == 'wasserstein':
                model_name = 'proto-%s-%s' % (args.distance_metric, args.cost_distance)
            else:
                model_name = 'proto_l2'

        plt.ylabel('|$\\Delta$ Label|')
        plt.xlabel('Distance')

        plt.title('Data: %s, Model: %s, n: %d, s_rho: %.3f, p_rho: %.3f' % (
            args.data, model_name, len(dist_list), s_rho, p_rho))
        plt.savefig('output/figs/%s.png' % model_name)
    return s_rho, p_rho

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-model_path', required=True)
    parser.add_argument('-model_type', required=True, choices=['gcn', 'proto'])
    args = parser.parse_args()

    model_type = args.model_type

    if model_type == 'gcn':
        model_class = GCN
    elif model_type == 'proto':
        model_class = ProtoNet
    else:
        assert False

    print('loading model from: %s' % args.model_path)

    model, args = load_model(args.model_path, model_class, None)
    test_data_loader = get_loader(
        args.data_dir, data_type='test', batch_size=BATCH_SIZE, shuffle=True,
        split=0)

    s_rho, p_rho = compute_r(model_type, model, args, test_data_loader)

if __name__ == '__main__':
    main()
