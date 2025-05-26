import copy
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from utils.repair import reset_bn_stats


def lerp(lam, t1, t2):
    t3 = copy.deepcopy(t1)
    for p in t1:
        t3[p] = (1 - lam) * t1[p] + lam * t2[p]
    return t3


def lerp_mult(params, lams=None):
    if lams is None:
        lams = np.ones(len(params)) / len(params)
    t1 = copy.deepcopy(params[0])
    for p in t1:
        t1[p] = t1[p] * lams[0]
        for lam, param in zip(lams[1:], params[1:]):
            t1[p] += lam * param[p]
    return t1


def linearly_combine(models, lams=None):
    if lams is None:
        lams = np.ones(len(models)) / len(models)

    device = list(models[0].parameters())[0].device
    with torch.no_grad():
        ret_model = copy.deepcopy(models[0]).to(device)
        x = parameters_to_vector(ret_model.parameters()) * lams[0]

        for model, lam in zip(models[1:], lams[1:]):
            y = parameters_to_vector(model.parameters())
            x += lam * y

        vector_to_parameters(x, ret_model.parameters())
    return ret_model


def merge_models(models, lams=None, reset_bn=False, loader=None):
    assert (reset_bn is False) or (reset_bn is True and loader is not None)
    conbined_model = linearly_combine(models, lams)

    if reset_bn is False:
        return conbined_model

    flag_bn = False
    for m in conbined_model.modules():
        if isinstance(m, nn.BatchNorm2d):
            flag_bn = True
            break

    if flag_bn is False:
        return conbined_model

    for p in conbined_model.modules():
        if isinstance(p, torch.nn.BatchNorm2d):
            p.track_running_stats = True
            p.running_mean = torch.zeros_like(p.weight)
            p.running_var = torch.ones_like(p.weight)
    conbined_model.cuda()
    reset_bn_stats(conbined_model, loader)
    return conbined_model
