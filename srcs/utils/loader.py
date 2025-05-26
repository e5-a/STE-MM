import torch


def load_model(net, dir, strict=True):
    ckpt = torch.load(dir, map_location=torch.device('cpu'))
    net.load_state_dict({'.'.join(k.split('.')[1:]): v
                         for k, v in ckpt['state_dict'].items()},
                        strict=strict)


def load_dm(name, batch_size, num_workers=8, seed=1, persistent_workers=True):
    from datasets import MNIST, CIFAR10, FMNIST
    return eval(name).DataModule(
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            seed=seed)


