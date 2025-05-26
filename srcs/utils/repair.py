import torch
import torch.nn as nn


def reset_bn_stats(model, loader, epochs=1):
    # resetting stats to baseline first as below is necessary for stability
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = None  # use simple average
            m.reset_running_stats()
    # run a single train epoch with augmentations to recalc stats
    device = list(model.parameters())[0].device
    model.train()
    for _ in range(epochs):
        with torch.no_grad():
            for images, _ in loader:
                output = model(images.to(device))

