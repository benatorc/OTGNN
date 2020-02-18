import torch
import torch.nn as nn

from graph import mol_features
import pdb


class GCN(nn.Module):

    def __init__(self, args):
        """Creates graph conv layers for molecular graphs."""
        super(GCN, self).__init__()
        self.args = args
        n_hidden = args.n_hidden

        self.n_atom_feats = mol_features.N_ATOM_FEATS
        self.n_bond_feats = mol_features.N_BOND_FEATS

        # Weights for the message passing network
        self.W_message_i = nn.Linear(self.n_atom_feats + self.n_bond_feats,
                                     n_hidden, bias=False,)
        self.W_message_h = nn.Linear(n_hidden, n_hidden, bias=False,)

        atom_n_hidden = args.pc_hidden
        if atom_n_hidden == -1:
            atom_n_hidden = n_hidden

        self.W_message_o = nn.Linear(self.n_atom_feats + n_hidden, atom_n_hidden)

        self.W_mol_h = nn.Linear(atom_n_hidden, args.n_ffn_hidden)
        self.W_mol_o = nn.Linear(args.n_ffn_hidden, args.n_labels)

        self.dropout_gcn = nn.Dropout(args.dropout_gcn)
        self.dropout_ffn = nn.Dropout(args.dropout_ffn)

        if args.batch_norm:
            self.batch_norm = nn.BatchNorm1d(n_hidden)

    def index_select_nei(self, input, dim, index):
        # Reshape index because index_select expects a 1-D tensor. Reshape the
        # output afterwards.
        target = torch.index_select(
            input=input,
            dim=dim,
            index=index.view(-1)
        )
        return target.view(index.size() + input.size()[1:])

    def aggregate_atoms(self, atom_h, scope):
        mol_h = []
        for (st, le) in scope:
            cur_atom_h = atom_h.narrow(0, st, le)

            if self.args.agg_func == 'sum':
                mol_h.append(cur_atom_h.sum(dim=0))
            elif self.args.agg_func == 'mean':
                mol_h.append(cur_atom_h.mean(dim=0))
            else:
                print('Aggregate function: %s not recognized' % (self.args.agg_func))
                assert(False)
        mol_h = torch.stack(mol_h, dim=0)
        return mol_h

    def forward(self, mol_graph):
        graph_inputs, scope = mol_graph.get_graph_inputs(
            device=self.args.device)
        fatoms, fbonds, agraph, bgraph = graph_inputs

        # nei_input_h is size [# bonds, n_hidden]
        nei_input_h = self.dropout_gcn(self.W_message_i(fbonds))
        # message_h is size [# bonds, n_hidden]
        message_h = nn.ReLU()(nei_input_h)

        for i in range(self.args.n_layers - 1):
            # nei_message_h is [# bonds, # max neighbors, n_hidden]
            nei_message_h = self.index_select_nei(
                input=message_h,
                dim=0,
                index=bgraph)

            # Sum over the nieghbors, now [# bonds, n_hidden]
            nei_message_h = nei_message_h.sum(dim=1)
            nei_message_h = self.dropout_gcn(self.W_message_h(nei_message_h))  # Shared weights -- m_{vw}^{t+1}

            message_h = nn.ReLU()(nei_input_h + nei_message_h)

        # Collect the neighbor messages for atom aggregation: [# atoms, # max neighbors, n_hidden]
        nei_message_h = self.index_select_nei(
            input=message_h,
            dim=0,
            index=agraph,
        )
        # Aggregate the messages
        nei_message_h = nei_message_h.sum(dim=1)  #  [# atoms, n_hidden]
        atom_input = torch.cat([fatoms, nei_message_h], dim=1)
        # atom_input = self.dropout(atom_input)

        # Atom embeddings
        atom_h = self.W_message_o(atom_input)
        if self.args.batch_norm:
            atom_h = self.batch_norm(atom_h)

        # Molecule embeddings
        if self.args.ffn_activation == 'ReLU':
            mol_h = nn.ReLU()(self.W_mol_h(self.aggregate_atoms(atom_h, scope)))
        elif self.args.ffn_activation == 'LeakyReLU':
            mol_h = nn.LeakyReLU()(self.W_mol_h(self.aggregate_atoms(atom_h, scope)))
        mol_o = self.W_mol_o(self.dropout_ffn(mol_h))
        return atom_h, mol_o
