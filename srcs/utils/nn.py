from torch.nn.utils import parameters_to_vector, vector_to_parameters
import torch
import types


def to_vec(arg):
    if isinstance(arg, torch.nn.Module):
        return parameters_to_vector(arg.parameters())
    if isinstance(arg, types.GeneratorType):
        return parameters_to_vector(arg)
    if isinstance(arg, torch.Tensor):
        return arg
    if isinstance(arg, dict):  # Maybe arg is the state_dict.
        return parameters_to_vector([v for key, v in arg.items() if not('running' in key or 'num_batches_tracked' in key)])
    assert True


