from collections import defaultdict
from typing import NamedTuple
from collections import OrderedDict

import torch
from scipy.optimize import linear_sum_assignment
import lap_uv
from analysis.metric import l2
import time
from sinkhorn.rebasinnet.graph.auto_graph import solve_graph
import numpy as np
#import lapjv


def generate_permutation_spec(model, input_shape, remove_nodes=list()):
    param_precision = next(iter(model.parameters())).data.dtype
    input = torch.randn(input_shape, dtype=param_precision)
    perm_dict, n_perm, permutation_g, parameter_map = solve_graph(
        model, input, remove_nodes=remove_nodes
    )
    #print(perm_dict, n_perm, permutation_g, parameter_map)
    #print(permutation_g.nodes.keys())
    #print(permutation_g.naming)

    map_param_index = dict()
    map_prev_param_index = dict()
    nodes = list(permutation_g.nodes.keys())
    for name, p in model.named_parameters():
        if parameter_map[name] not in nodes:
            continue
        map_param_index[name] = permutation_g.naming[parameter_map[name]]
        parents = permutation_g.parents(parameter_map[name])
        map_prev_param_index[name] = (
            None if len(parents) == 0 else permutation_g.naming[parents[0]]
        )

    #print(map_param_index)
    #print(map_prev_param_index)

    axes_to_perm = {}
    for name, p in model.named_parameters():
        if (name not in map_param_index or
                name not in map_prev_param_index):
            continue

        output_index = perm_dict[map_param_index[name]]
        input_index = (
            perm_dict[map_prev_param_index[name]]
            if map_prev_param_index[name] is not None
            else None
        )
        if 'bias' in name[-4:]:
            if output_index is not None:
                axes_to_perm[name] = (f'P_{output_index}', None)
        elif len(p.shape) == 1:
            if output_index is not None:
                axes_to_perm[name] = (f'P_{output_index}', None)
        elif isinstance(dict(model.named_modules())['.'.join(name.split('.')[:-1])], torch.nn.LayerNorm):
            if output_index is not None:
                axes_to_perm[name] = (f'P_{output_index}', None)
        elif 'weight' in name[-6:]:
            P_in = f'P_{input_index}' if input_index is not None else None
            P_out = f'P_{output_index}' if output_index is not None else None
            axes_to_perm[name] = (P_out, P_in)

    for name, m in model.named_modules():
        if "BatchNorm" in str(type(m)):
            if name + ".weight" in map_param_index:
                if m.running_mean is None and m.running_var is None:
                    continue
                i = perm_dict[map_param_index[name + ".weight"]]
                axes_to_perm[name + '.running_mean'] = (f'P_{i}', None)
                axes_to_perm[name + '.running_var'] = (f'P_{i}', None)
    #return axes_to_perm
    return permutation_spec_from_axes_to_perm(axes_to_perm)



class PermutationSpec(NamedTuple):
    perm_to_axes: dict
    axes_to_perm: dict


def permutation_spec_from_axes_to_perm(axes_to_perm: dict) -> PermutationSpec:
    perm_to_axes = defaultdict(list)
    for wk, axis_perms in axes_to_perm.items():
        for axis, perm in enumerate(axis_perms):
            if perm is not None:
                perm_to_axes[perm].append((wk, axis))
    return PermutationSpec(perm_to_axes=dict(perm_to_axes), axes_to_perm=axes_to_perm)


def get_permuted_param(ps: PermutationSpec, perm, k: str, params, except_axis=None):
    """Get parameter `k` from `params`, with the permutations applied."""
    w = params[k]

    if k not in ps.axes_to_perm:
        return w
    # if k == 'classifier.weight':  # to reshape because of input shape is 3x 96 x 96
    #     w = w.reshape(126, 512 * 4, 3, 3)
    for axis, p in enumerate(ps.axes_to_perm[k]):
        # Skip the axis we're trying to permute.
        if axis == except_axis:
            continue

        # None indicates that there is no permutation relevant to that axis.
        if p is not None:
            w = torch.index_select(w, axis, perm[p].int())
    # if k == 'classifier.weight':
    #     w = w.reshape(126, -1)
    return w


def apply_permutation(ps: PermutationSpec, perm, params):
    """Apply a `perm` to `params`."""
    ret = {}
    for k in params.keys():
        if params[k].dim() != 0:  # avoid num_batches_tracked
            ret[k] = get_permuted_param(ps, perm, k, params)
        else:
            ret[k] = params[k]
    return ret


class weight_matching:
    def __init__(self,
                 ps: PermutationSpec,
                 verbose: bool = False):
        self.verbose = verbose
        self.ps = ps
        self.perm = None

    def fit(self,
            params_a,
            params_b,
            max_iter=300):
        self.params_a = {key: v.cpu() for key, v in params_a.items()}
        self.params_b = {key: v.cpu() for key, v in params_b.items()}
        device = list(params_a.values())[0].device
        perm_sizes = {p: self.params_a[axes[0][0]].shape[axes[0][1]]
                      for p, axes in self.ps.perm_to_axes.items()}
        self.perm = {p: torch.arange(n) for p, n in perm_sizes.items()} \
            if self.perm is None else self.perm
        self.perm = {key: self.perm[key].cpu() for key in self.perm}  # to cpu

        perm_names = list(self.perm.keys())
        self.metrics = {}
        for iteration in range(max_iter):
            progress = False
            for p_ix in torch.randperm(len(perm_names)):
                start_time = time.time()

                p = perm_names[p_ix]
                n = perm_sizes[p]
                A = torch.zeros((n, n))
                for wk, axis in self.ps.perm_to_axes[p]:  # layer loop
                    if ('running_mean' in wk) or ('running_var' in wk) \
                            or ('num_batches_tracked' in wk):
                        continue
                    A += self.cost(wk, axis, n)
                start_lap = time.time()
                ri, ci = linear_sum_assignment(
                        A.detach().numpy(), maximize=True)

                #ci = np.zeros(n, dtype=np.int32)
                #ri = np.zeros(n, dtype=np.int32)
                #u = np.zeros(n, dtype=np.float32)
                #v = np.zeros(n, dtype=np.float32)
                #cost = lap_uv.lap(-A.detach().numpy(), ci, ri, u, v)

                #ci, ri, _ = lapjv.lapjv(-A.detach().numpy())
                lap_time = time.time() - start_lap
                #assert (torch.tensor(ri) == torch.arange(len(ri))).all()
                oldL = torch.einsum('ij,ij->i', A,
                                    torch.eye(n)[self.perm[p].long()]).sum()
                newL = torch.einsum('ij,ij->i', A, torch.eye(n)[ci, :]).sum()
                if self.verbose:
                    print(f"{iteration}/{p}: {newL - oldL} {(newL - oldL)/oldL}")
                #print(f"{lap_time} / {time.time() - start_time}")
                #progress = progress or newL > oldL + 1e-12
                progress = progress or not(np.isclose(newL.item(), oldL.item()))
                #progress = progress or (((newL - oldL)/oldL) > 0.1)
                self.perm[p] = torch.Tensor(ci)
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
