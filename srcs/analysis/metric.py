import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.nn import to_vec
import copy


def test(model, test_loader, verbose=True):
    model.eval()
    device = list(model.parameters())[0].device
    test_loss = 0
    correct = 0

    len_dataset = 0
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            for data, target in tqdm(test_loader, disable=not(verbose)):
                data, target = data.to(device), target.to(device)
                output = model(data)
                len_dataset += len(data)
                output = F.log_softmax(output, dim=1)

                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len_dataset
    acc = 100. * correct / len_dataset
    return test_loss, acc


def l2(net_a, net_b):
    params_a = to_vec(net_a)
    params_b = to_vec(net_b)
    return torch.linalg.vector_norm(params_a - params_b).detach().item()


def rademacher(shape, dtype=torch.float32, device='cuda:0'):
    rand = ((torch.rand(shape, device=device) < 0.5)) * 2. - 1.
    return rand.to(dtype)


def set_attr(obj, names, val):
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        set_attr(getattr(obj, names[0]), names[1:], val)


def load_weights(mod, names, params):
    for name, p in zip(names, params):
        set_attr(mod, name.split("."), p)


def del_attr(obj, names):
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:])


def make_functional(mod):
    orig_params = tuple(mod.parameters())
    # Remove all the parameters in the model
    names = []
    for name, p in list(mod.named_parameters()):
        del_attr(mod, name.split("."))
        names.append(name)
    return orig_params, names


def hutchinson_trace_estimator(
        model,
        data_loader,
        weighted=False,
        num_epochs=1,
        create_graph=False
        ):
    device = list(model.parameters())[0].device
    dup_model = copy.deepcopy(model)

    params, names = make_functional(dup_model)
    params = tuple([p.detach().requires_grad_() for p in params])

    def calc(data, target):
        data, target = data.to(device), target.to(device)

        def subfunc(*params):
            load_weights(dup_model, names, params)
            dup_model.eval()
            with torch.set_grad_enabled(True):
                output = dup_model(data)
                output = F.log_softmax(output, dim=1)

                loss = F.nll_loss(output, target, reduction='sum')

            loss /= len(data)
            return loss
        return subfunc

    count = 0
    loss_sum = 0
    ret_sum = 0
    for epoch in range(num_epochs):
        for data, target in tqdm(data_loader):
            count += 1
            with torch.no_grad():
                if weighted is False:
                    v = tuple([rademacher(p.shape) for p in model.parameters()])
                else:
                    v = tuple([rademacher(p.shape) * torch.abs(p) for p in model.parameters()])
            loss, ret = torch.autograd.functional.vhp(
                    calc(data, target), params,
                    v=v, create_graph=create_graph, strict=True)
            for vh, vp in zip(ret, v):
                ret_sum += torch.dot(vh.reshape(-1), vp.reshape(-1)).item()
            loss_sum += loss.item()
    ret_sum /= count
    loss_sum /= count
    return loss_sum, ret_sum
