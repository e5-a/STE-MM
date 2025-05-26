from search.weight_matching import weight_matching, generate_permutation_spec
from search.weight_matching_uv import weight_matching_uv
from utils.nn import to_vec
from torch.nn.utils import vector_to_parameters
import copy
import numpy as np
import logging

log = logging.getLogger(__name__)


def match_wm(nets, input_shape, fast_wm=False, verbose=True):
    for net in nets:
        net.cuda()

    ps = generate_permutation_spec(nets[0],
                                   [1, *input_shape])
    if fast_wm is True:
        wms = [weight_matching_uv(ps) for _ in nets]
    else:
        wms = [weight_matching(ps) for _ in nets]

    ret_nets = []
    for net in nets:
        ret_nets.append(copy.deepcopy(net))
    n = len(nets)
    net = copy.deepcopy(nets[0])

    old_l2_dists = np.ones(len(nets)) * 1e10
    new_l2_dists = np.zeros(len(nets))
    count = 0
    while True:
        if verbose is True:
            log.info(f'count: {count}')
        ll = list(range(n))
        np.random.shuffle(ll)
        for l in ll:
            targets = [i for i in range(n) if i != l]
            v = to_vec(ret_nets[targets[0]]) / (n-1)
            for t in targets[1:]:
                v += to_vec(ret_nets[t]) / (n-1)
            vector_to_parameters(v, net.parameters())
            wms[l].fit(net.state_dict(), nets[l].state_dict())
            #print(wms[l].metrics)
            new_l2_dists[l] = wms[l].metrics['l2_dist'][-1]
            ret_nets[l] = copy.deepcopy(nets[l])
            wms[l].transform(ret_nets[l])
        if verbose is True:
            log.info(f'{new_l2_dists}')
            log.info(f'{np.sum(new_l2_dists)} {np.sum(old_l2_dists)}')
        #if (np.sum(new_l2_dists) > (np.sum(old_l2_dists) + 1e-4)):
        if np.isclose(np.sum(new_l2_dists), np.sum(old_l2_dists)):
            break
        old_l2_dists = copy.deepcopy(new_l2_dists)
        count += 1
    return ret_nets, wms
