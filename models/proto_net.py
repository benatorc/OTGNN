import torch
import torch.nn as nn
import numpy as np
import random
import ot
from collections import defaultdict
import rdkit.Chem as Chem
import math

from graph import MolGraph
from .gcn import GCN
from .ot_modules import compute_ot, compute_cost_mat

import pdb

class ProtoNet(nn.Module):
    def __init__(self, args, init_model=None):
        super(ProtoNet, self).__init__()
        self.args = args

        # if no pretrained model supplied, instantiate a new gcn model
        if init_model is not None:
            self.gcn = init_model
        else:
            self.gcn = GCN(args)

        # Hacky way to get param count correct
        self.gcn.W_mol_h.weight.requires_grad = False
        self.gcn.W_mol_o.weight.requires_grad = False

        self.pc_list = None  # List of trainable point clouds
        self.h_list = []

        n_pc, pc_size = args.n_pc, args.pc_size
        self.pc_source = args.init_method

        if args.init_method != 'data':
            self.pc_list = nn.ParameterList([])
            for idx in range(n_pc):
                if args.distance_metric in ['wasserstein']:
                    pc_X = nn.Parameter(torch.empty([pc_size, args.pc_hidden],
                                                    requires_grad=True,
                                                    device=args.device))
                    nn.init.xavier_normal_(pc_X, gain=args.pc_xavier_std)
                    pc_H = np.ones([pc_size]) / pc_size

                    self.pc_list.append(pc_X)
                    self.h_list.append(pc_H)
                elif args.distance_metric in ['l2', 'dot']:
                    pc_X = nn.Parameter(torch.empty([1, args.pc_hidden], requires_grad=True, device=args.device))
                    nn.init.xavier_normal_(pc_X)
                    self.pc_list.append(pc_X)
                else:
                    print('Distance metric: %s not recognized' % args.distance_metric)
                    assert(False)
        else:
            for idx in range(n_pc):
                smiles = args.pc_data[0][idx]
                mol = Chem.MolFromSmiles(smiles)
                n_atoms = mol.GetNumAtoms()

                pc_H = np.ones([n_atoms]) / n_atoms
                self.h_list.append(pc_H)


        # Reduce dimensionality of GCN atom embeddings
        # self.W_atom_d = nn.Linear(args.n_hidden, args.pc_hidden)

        self.dropout_ffn = nn.Dropout(args.dropout_ffn)

        n_input = n_pc
        # Weights for the computed distance
        self.W_dist_h = nn.Linear(n_input, args.n_ffn_hidden)
        self.W_out = nn.Linear(args.n_ffn_hidden, args.n_labels)

    def get_model_optim(self):
        pc_param_names = []
        for name, _ in self.pc_list.named_parameters():
            pc_param_names.append('pc_list.' + name)  # hack

        pc_params_list = []
        non_pc_params_list = []

        all_names = []

        for name, param in self.named_parameters():
            all_names.append(name)
            if name in pc_param_names:
                pc_params_list.append(param)
            else:
                non_pc_params_list.append(param)

        optimizer = torch.optim.Adam([
            {'params': pc_params_list, 'lr': self.args.lr_pc},
            {'params': non_pc_params_list}], lr=self.args.lr,)
        return optimizer

    def get_graph_pc_embeddings(self, mol_graph):
        # Returns a list of point cloud embeddings of given input mol_graphs
        atom_h, _ = self.gcn(mol_graph)
        # atom_h = self.W_atom_d(atom_o)

        scope = mol_graph.scope

        pc_list = []
        for (st, le) in scope:
            pc_X = atom_h.narrow(0, st, le)

            if self.args.distance_metric in ['l2', 'dot']:
                if self.args.agg_func == 'sum':
                    pc_X = torch.sum(pc_X, dim=0, keepdim=True)
                elif self.args.agg_func == 'mean':
                    pc_X = torch.mean(pc_X, dim=0, keepdim=True)
                else:
                    assert False

            pc_list.append(pc_X)
        return pc_list

    def get_avg_pc_grad_norm(self):
        sum_grad_norm = 0
        for pc in self.pc_list:
            norm = torch.norm(pc.grad, p=2)
            sum_grad_norm += norm.item()
        return sum_grad_norm / self.args.n_pc

    def forward(self, mol_graph, debug=False):
        # Use GCN to compute atom embeddings
        # atom_o is [# atoms, # features], output of gcn
        # atom_h is transformed version that downsizes the hidden dim
        atom_h, _ = self.gcn(mol_graph)

        # scope delienates the boundaries of the molecules in atom_h
        scope = mol_graph.scope

        # Stores the model output values
        output_dict = defaultdict()

        if self.args.init_method == 'data':
            self.pc_list = self.get_graph_pc_embeddings(MolGraph(self.args.pc_data[0]))

        # Keep track of the distances to point clouds
        mol_pc_dist = []
        nce_reg_list = []
        for mol_idx, (st, le) in enumerate(scope):

            # Fetch the atoms for the corresponding molecular graph
            atom_X = atom_h.narrow(0, st, le)

            # Construct uniform prob dist over the atoms for this molecule
            atom_H = np.ones([le]) / le

            cur_dist_list = []
            for pc_idx in range(self.args.n_pc):
                pc_X = self.pc_list[pc_idx]

                if self.args.distance_metric == 'wasserstein':
                    ot_dist, nce_reg, ot_mat, cost_mat = compute_ot(
                        X_1=pc_X, X_2=atom_X, H_1=self.h_list[pc_idx], H_2=atom_H,
                        device=self.args.device, opt_method=self.args.opt_method,
                        sinkhorn_entropy=self.args.sinkhorn_entropy,
                        sinkhorn_max_it=self.args.sinkhorn_max_it,
                        cost_distance=self.args.cost_distance,
                        unbalanced=self.args.unbalanced,
                        nce_coef=self.args.nce_coef)

                    if self.args.nce_coef > 0:
                        nce_reg_list.append(nce_reg)

                    if self.args.mult_num_atoms:
                        # Divide by 100 to avoid large gradients
                        ot_dist = ot_dist * pc_X.size()[0] * atom_X.size()[0] / 100

                    cur_dist_list.append(ot_dist)
                elif self.args.distance_metric == 'l2':
                    if self.args.agg_func == 'sum':
                        mol_X = torch.sum(atom_X, dim=0)
                    elif self.args.agg_func == 'mean':
                        mol_X = torch.mean(atom_X, dim=0)
                    else:
                        assert False
                    l2_dist = torch.sum((mol_X - pc_X.squeeze(0)) ** 2)
                    cur_dist_list.append(l2_dist)
                elif self.args.distance_metric == 'dot':
                    if self.args.agg_func == 'sum':
                        mol_X = torch.sum(atom_X, dim=0)
                    elif self.args.agg_func == 'mean':
                        mol_X = torch.mean(atom_X, dim=0)
                    else:
                        assert False
                    dot_dist = torch.sum(mol_X * pc_X.unsqueeze(0))
                    cur_dist_list.append(dot_dist)
                else:
                    assert(False)

            cur_dist_list = torch.stack(cur_dist_list)
            mol_pc_dist.append(cur_dist_list)

            # pc_1 = self.pc_list[0]
            # output_dict['norm_pc1'] += norm(pc_1)
        # mol_pc_dist [batch_size, num_point_clouds]
        mol_pc_dist = -1 * torch.stack(mol_pc_dist)

        # print(mol_pc_dist)
        if self.args.ffn_activation == 'ReLU':
            mol_dist_h = nn.ReLU()(self.W_dist_h(mol_pc_dist))
        elif self.args.ffn_activation == 'LeakyReLU':
            mol_dist_h = nn.LeakyReLU(0.2)(self.W_dist_h(mol_pc_dist))
        else:
            assert(False)
        # print(mol_pc_dist)
        # print(mol_dist_h)
        preds = self.W_out(self.dropout_ffn(mol_dist_h))
        # print(preds)

        if self.args.nce_coef > 0:
            nce_reg = torch.mean(torch.stack(nce_reg_list))
        else:
            nce_reg = None

        output_dict.update(
            {'preds': preds, 'atom_h': atom_h, 'nce_reg': nce_reg})

        if debug:
            pdb.set_trace()

        return output_dict
