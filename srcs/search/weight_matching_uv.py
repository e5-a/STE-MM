from collections import defaultdict
from typing import NamedTuple
from collections import OrderedDict

import torch
from scipy.optimize import linear_sum_assignment
from analysis.metric import l2
import time
from sinkhorn.rebasinnet.graph.auto_graph import solve_graph
import numpy as np
# from pylibraft.solver import lap
# import lapjv
import lap_uv


from search.weight_matching import PermutationSpec, apply_permutation, get_permuted_param


class weight_matching_uv:
    def __init__(self,
                 ps: PermutationSpec,
                 verbose: bool = False):
        self.verbose = verbose
        self.ps = ps
        self.perm = None

    def init_uv(self, perm_sizes):
        self.u = {}
        self.v = {}
        perm_names = list(self.perm.keys())
        for i in range(len(self.perm.keys())):
            n = perm_sizes[perm_names[i]]
            self.u[i] = np.zeros(n, dtype=np.float32)
            self.v[i] = np.zeros(n, dtype=np.float32)

    def fit(self,
            params_a,
            params_b,
            max_iter=300):
        device = list(params_a.values())[0].device
        self.params_a = {key: v for key, v in params_a.items()}
        self.params_b = {key: v for key, v in params_b.items()}
        perm_sizes = {p: self.params_a[axes[0][0]].shape[axes[0][1]]
                      for p, axes in self.ps.perm_to_axes.items()}
        self.perm = {p: torch.arange(n).to(device) for p, n in perm_sizes.items()} \
            if self.perm is None else self.perm
        if not hasattr(self, 'u'):
            self.init_uv(perm_sizes)
        # self.perm = {key: self.perm[key] for key in self.perm}  # to cpu

        perm_names = list(self.perm.keys())
        self.metrics = {}
        for iteration in range(max_iter):
            progress = False
            for p_ix in torch.randperm(len(perm_names)):
                start_time = time.time()

                p = perm_names[p_ix]
                n = perm_sizes[p]
                A = torch.zeros((n, n)).to(device)
                for wk, axis in self.ps.perm_to_axes[p]:  # layer loop
                    if ('running_mean' in wk) or ('running_var' in wk) \
                            or ('num_batches_tracked' in wk):
                        continue
                    A -= self.cost(wk, axis, n)
                B = A.detach().cpu().numpy()

                B -= self.v[p_ix.item()]
                B = (B.T - self.u[p_ix.item()]).T
                #B -= np.min(B, axis=0)

                ci = np.zeros(n, dtype=np.int32)
                ri = np.zeros(n, dtype=np.int32)
                u = np.zeros(n, dtype=np.float32)
                v = np.zeros(n, dtype=np.float32)
                cost = lap_uv.lap(B, ci, ri, u, v)
                self.u[p_ix.item()] += u
                self.v[p_ix.item()] += v

                ci = torch.from_numpy(ci).to(device)
                identity = torch.eye(n).to(device)
                oldL = -torch.einsum('ij,ij->i', A,
                                     identity[self.perm[p].long()]).sum()
                newL = -torch.einsum('ij,ij->i', A, identity[ci, :]).sum()
                if self.verbose:
                    print(f"{iteration}/{p}: {newL - oldL}, {cost}")
                progress = progress or not(np.isclose(newL.item(), oldL.item()))
                self.perm[p] = ci
            self.distance()
            if not progress:
                break
        self.perm = {key: self.perm[key].to(device) for key in self.perm}

    def cost(self, wk, axis, n):
        w_a = self.params_a[wk]  # target
        w_b = get_permuted_param(self.ps, self.perm, wk, self.params_b, except_axis=axis) 
        #if len(w_a.shape) == 2:
        #    w_a = w_a @ w_a.T @ w_a
        #    w_b = w_b @ w_b.T @ w_b
        w_a = torch.moveaxis(w_a, axis, 0).reshape((n, -1))
        w_b = torch.moveaxis(w_b, axis, 0).reshape((n, -1))
        return w_a @ w_b.T  # A is cost matrix to assignment,

    def distance(self):
        p_params_b = apply_permutation(self.ps, self.perm, self.params_b)
        l2_dist = l2(self.remove_bn(self.params_a), self.remove_bn(p_params_b))
        if self.verbose is True:
            print(l2_dist)
        self.append_metric('l2_dist', l2_dist)

    def remove_bn(self, params):
        def contains_any(string):
            return any(substr in string for substr in ['running_mean', 'running_var', 'num_batches_tracked'])
        return {key: v for (key, v) in params.items() if not contains_any(key)}

    def transform(self, model):
        model.load_state_dict(
                apply_permutation(self.ps, self.perm, model.state_dict()))

    def append_metric(self, name, value):
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)

    def save(self, filename):
        torch.save(self.perm, filename)

    def load(self, arg):
        if isinstance(arg, str):
            self.perm = torch.load(arg)
        elif isinstance(arg, dict):
            self.perm = arg
        self.perm = {key: self.perm[key].cpu() for key in self.perm}  # to cpu


class StoreBN(object):
    def __init__(self):
        self.full_key = None
        self.state_dict = None

    def remove_bn(self, params):
        self.full_key = list(params.keys())
        self.state_dict = OrderedDict()  # init
        for key in self.full_key:
            if ('running_var' in key) or ('running_mean' in key):
                self.state_dict[key] = params[key]
                del params[key]
        return params

    def repair_bn(self, params):
        for key in self.full_key:
            if ('running_var' in key) or ('running_mean' in key):
                params[key] = self.state_dict[key]
            else:
                params[key] = params[key]
        return params
