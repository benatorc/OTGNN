import torch
import ot
import numpy as np

import pdb


def compute_cost_mat(X_1, X_2, rescale_cost=False, cost_distance='l2'):
    '''Computes the l2 cost matrix between two point cloud inputs.

    Args:
        X_1: [#nodes_1, #features] point cloud, tensor
        X_2: [#nodes_2, #features] point cloud, tensor
        rescale_cost: Whether to normalize the cost matrix by the max ele

    Output:
        [#nodes_1, #nodes_2] matrix of the l2 distance between point pairs
    '''
    n_1, _ = X_1.size()
    n_2, _ = X_2.size()

    if cost_distance == 'l2':
        # Expand dimensions to broadcast in the following step
        X_1 = X_1.view(n_1, 1, -1)
        X_2 = X_2.view(1, n_2, -1)
        squared_dist = (X_1 - X_2) ** 2
        cost_mat = torch.sum(squared_dist, dim=2)
    elif cost_distance == 'dot':
        cost_mat = - X_1.matmul(X_2.transpose(0,1))
    else:
        assert False

    if rescale_cost:
        cost_mat = cost_mat / cost_mat.max()
    return cost_mat


def compute_ot(X_1, X_2, H_1, H_2, sinkhorn_entropy, device='cpu',
               opt_method='sinkhorn_stabilized', rescale_cost=False,
               sinkhorn_max_it=5, cost_distance='l2', unbalanced=False,
               nce_coef=0):
    ''' Computes the optimal transport distance

    Args:
        X_1: [#nodes_1, #features] point cloud, tensor
        X_2: [#nodes_2, #features] point cloud, tensor
        H_1: [#nodes_1] histogram, numpy array
        H_2: [#nodes_2] histogram, numpy array
        device: cpu or gpu
        opt_method: The optimization method
        sinkhorn_entropy: Entropy regularizer for sinkhorn
        rescale_cost: Whether to normalize the cost matrix by the max ele
        sinkhorn_max_iters: the maximum number of iterations to run sinkhorn
    '''
    cost_mat = compute_cost_mat(
        X_1, X_2, rescale_cost=rescale_cost, cost_distance=cost_distance)

    # Convert cost matrix to numpy array to feed into sinkhorn algorithm
    cost_mat_detach = cost_mat.detach().cpu().numpy()
    if unbalanced:
        if opt_method != 'emd':
            ot_mat = ot.unbalanced.sinkhorn_unbalanced(
                a=H_1, b=H_2, M=cost_mat_detach,
                numItermax=sinkhorn_max_it,
                reg=sinkhorn_entropy,
                reg_m=1.,
                method=opt_method)
        else:
            assert False, "opt_method emd uncompatible with unbalanced OT"
    else:
        if opt_method == 'emd':
            ot_mat = ot.emd(a=H_1, b=H_2, M=np.max(np.abs(cost_mat_detach)) + cost_mat_detach, numItermax=sinkhorn_max_it)
        else:
            ot_mat = ot.sinkhorn(a=H_1, b=H_2, M=cost_mat_detach,
                            numItermax=sinkhorn_max_it,
                            reg=sinkhorn_entropy,
                            method=opt_method)

    ot_mat_attached = torch.tensor(ot_mat, device=device, requires_grad=False).float()
    ot_dist = torch.sum(ot_mat_attached * cost_mat)

    nce_reg = 0
    if nce_coef > 0:
        all_nce_dists = []
        all_nce_dists.append(- ot_dist)
        for _ in range(5):
            random_mat = ot_mat.copy()
            np.random.shuffle(random_mat)
            random_mat = torch.tensor(random_mat, device=device, requires_grad=False).float()
            all_nce_dists.append(- torch.sum(random_mat * cost_mat))

        for _ in range(5):
            random_mat = np.random.rand(H_1.shape[0], H_2.shape[0]) * 10.
            while np.linalg.norm(np.sum(random_mat, axis=0) - H_2) > 1e-3 and np.linalg.norm(np.sum(random_mat, axis=1) - H_1) > 1e-3:
                random_mat = random_mat / np.sum(random_mat, axis=0, keepdims=True) * H_2.reshape((1, H_2.shape[0]))
                random_mat = random_mat / np.sum(random_mat, axis=1, keepdims=True) * H_1.reshape((H_1.shape[0], 1))
            random_mat = torch.tensor(random_mat, device=device, requires_grad=False).float()
            all_nce_dists.append(- torch.sum(random_mat * cost_mat))

        nce_reg = torch.nn.LogSoftmax()(torch.stack(all_nce_dists))[0]

    return ot_dist, nce_reg, ot_mat_attached, cost_mat
