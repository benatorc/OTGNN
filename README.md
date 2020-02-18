# Mol OT

Dependencies required:
```
conda create -n mol_ot python=3.6.8
sudo apt-get install libxrender1

conda install pytorch torchvision -c pytorch
conda install -c rdkit rdkit
conda install -c conda-forge pot
conda install -c anaconda scikit-learn
conda install -c conda-forge matplotlib
conda install -c conda-forge tqdm
conda install -c conda-forge tensorboardx
```

---


Relevant parameters:

| Arg      | Description |
| ----------- | ----------- |
| n_layers | Num layers for GCN |
| n_hidden | Hidden dimension for GCN |
| n_ffn_hidden | Num FFN for GCN |
| dropout_gcn | Dropout for GCN |
| dropout_ffn | Dropout for FFN |
| agg_func | choices = ['sum', 'mean'] |
| distance_metric | choices = ['wasserstein', 'l2', 'dot', ] |
| n_pc | Number of point clouds (prototypes) |
| pc_size | Number of nodes in each point cloud |
| pc_hidden | Hidden dimension of the nodes in the point clouds |
| ffn_activation | choices = ['ReLU', 'LeakyReLU'] |
| mult_num_atoms | Whether to multiply Wasserstein distances by number of points in prototype point cloud and graph point cloud |
| opt_method | choices = ['sinkhorn', 'sinkhorn_stabilized', 'emd', 'greenkhorn', 'sinkhorn_epsilon_scaling']|
| cost_distance | choices = ['l2', 'dot'] |
| nce_coef | The lambda for noise contrastive regularizer |
| log_tb   | Whether to log the results to tensorboard, you can open tensorboard results using the command (assuming all your output dirs are in a folder called "output"): 'tensorboard --logdir output'      |
| n_splits   | Number of splits to train the data on |

Note that training no longer happens on the first epoch, so that you can log stats before any training is done.

---
### Graph Representation in code

Each batch of graph input is abstracted as a "MolGraph" object, which is fed to the GCN to compute atom embeddings.
The output of the GCN is a 2-D atom embedding matrix for each batch of graph input (MolGraph object).
The "scope" variable in the MolGraph object marks the boundaries of the atoms in each graph so you can use the "narrow" function to get back the atom embedding for each specific graph.

---

### Directories

| Directory | Description |
| --- | ----------- |
| data | Contains the property datasets |
| dataset | Wrapper for data inputs/loaders |
| graph | Parses each batch of smiles string to create MolGraph objects that computes the input tensors for the model. |
| models | graph conv/prototype models |
| utils | Various data/write utils |

---


### Sample code

Run GCN code:

```
python train_gcn.py -data sol -output_dir output/test -n_epochs 150 -lr 5e-4 \
  -n_hidden 50 -n_ffn_hidden 100 -agg_func sum -batch_size 16
```

Run Wasserstein prototype:

```
python train_proto.py -data sol -output_dir output/test -lr 5e-4 \
  -n_epochs 150 -n_hidden 50 -n_ffn_hidden 100 -batch_size 16 -n_pc 10 \
  -pc_size 10 -pc_hidden 10 -distance_metric wasserstein -separate_lr \
  -lr_pc 5e-3 -opt_method emd -mult_num_atoms -nce_coef 0.01
```
