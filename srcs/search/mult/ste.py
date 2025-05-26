from collections import OrderedDict
import torch
import torchopt
from tqdm import tqdm
import numpy as np
from search.weight_matching_uv import weight_matching_uv
from search.weight_matching import generate_permutation_spec, weight_matching
from torch.nn.utils.stateless import functional_call
import torch.nn.functional as F
from utils.nn import to_vec
from utils.merge import lerp_mult, merge_models
from analysis.metric import test, l2
import copy
import time
#from multiprocessing import Pool
#import multiprocessing as mp


def flatten_states(states):
    res = []
    for state in states:
        res.extend(state.values())
    return res


def unflatten_states(states, refs):
    res = []
    i = 0
    for ref in refs:
        res.append(OrderedDict())
        for key in ref:
            res[-1][key] = states[i]
            i += 1
    return res


def matching(wm, tgt, org):
    wm.fit(tgt, org)
    return wm


def cos_sim(v_a, v_b):
    return torch.sum(v_a * v_b) / torch.linalg.norm(v_a) / torch.linalg.norm(v_b)

def ste(model_a,
        models,
        train_loader,
        test_loader,
        learning_rate,
        num_epochs,
        mid,
        device,
        input_shape,
        fast_wm):

    assert mid in [True, False]
    #mp.set_start_method('spawn')
    train_states = [copy.deepcopy(model_a.state_dict()) for _ in models]
    ps = generate_permutation_spec(model_a, [1, *input_shape])
    if fast_wm is True:
        wms = [weight_matching_uv(ps) for _ in models]
    else:
        wms = [weight_matching(ps) for _ in models]
    optimizer = torchopt.adam(lr=learning_rate)
    opt_state = optimizer.init(flatten_states(train_states))

    for p in model_a.parameters():
        p.requires_grad = False

    for epoch in tqdm(range(num_epochs)):
        processing_times = []
        correct = 0
        sum_loss = 0
        n_data = 0
        with tqdm(total=len(train_loader)) as pbar:
            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)
                n_data += len(x)

                start_time = time.time()
                projected_models = []
                for wm, model, state in zip(wms, models, train_states):
                    wm.fit(state, model.state_dict())
                    projected_models.append(copy.deepcopy(model))
                    wm.transform(projected_models[-1])
                end_time = time.time()
                processing_times.append(end_time - start_time)
                pbar.set_postfix(avg_time=f"{np.mean(processing_times):.4f} s")

                for state in train_states:
                    for key in state:
                        if 'running' in key or 'num_batches_tracked' in key:
                            continue
                        #print(key)
                        state[key] = state[key].detach()
                        state[key].requires_grad = True
                        state[key].grad = None


                ste_states = []
                for i, projected_model in enumerate(projected_models):
                    ste_states.append({})
                    for key in projected_model.state_dict():
                        ste_states[i][key] = \
                                projected_model.state_dict()[key].detach() + (
                                        train_states[i][key] - train_states[i][key].detach())

                if mid is False:
                    lams = np.random.rand(len(ste_states)+1)
                else:
                    lams = np.ones(len(ste_states)+1)
                lams /= np.sum(lams)
                midpoint_params = lerp_mult([model_a.state_dict(), *ste_states], lams)

                model_a.train()
                output = functional_call(model_a, midpoint_params, x)
                loss = F.cross_entropy(output, y)
                loss.backward()
                sum_loss += loss.item()

                with torch.no_grad():
                    grads_of_models = []
                    for state in train_states:
                        grads_of_models.append({})
                        grads = grads_of_models[-1]
                        for key in state:
                            if state[key].grad is None:
                                grads[key] = torch.zeros_like(state[key])
                            else:
                                grads[key] = state[key].grad
                    updates, opt_state = optimizer.update(
                            flatten_states(grads_of_models), opt_state, params=flatten_states(train_states), inplace=False)
                    tmp_states = torchopt.apply_updates(
                            flatten_states(train_states), updates, inplace=False)
                    train_states = unflatten_states(tmp_states, train_states)
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(y.view_as(pred)).sum().item()

                pbar.update(1)

        acc = correct / n_data
        print('acc/train:', acc, epoch)
        print('loss/train:', sum_loss / n_data, epoch)

        permed_nets = [model_a]
        for net, w in zip(models, wms):
            new_net = copy.deepcopy(net)
            w.transform(new_net)
            permed_nets.append(new_net)
    return permed_nets, wms
