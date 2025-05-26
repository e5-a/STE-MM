from .resnet_plus import ResNet, Bottleneck
from torchvision.models import resnet50
import torch
import torch.nn.functional as F
import numpy as np


class BlurPoolConv2d(torch.nn.Module):
    def __init__(self, conv):
        super().__init__()
        default_filter = torch.tensor([[[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]]) / 16.0
        filt = default_filter.repeat(conv.in_channels, 1, 1, 1)
        self.conv = conv
        self.register_buffer('blur_filter', filt)

    def forward(self, x):
        blurred = F.conv2d(x, self.blur_filter, stride=1, padding=(1, 1),
                           groups=self.conv.in_channels, bias=None)
        return self.conv.forward(blurred)


def Net(w=1, num_classes=10, **kwargs):
    #return ResNet(Bottleneck, [3, 4, 6, 3], w=w, num_classes=num_classes)

    model = resnet50(weights=None, num_classes=num_classes)
    #def apply_blurpool(mod: torch.nn.Module):
    #    for (name, child) in mod.named_children():
    #        if isinstance(child, torch.nn.Conv2d) and (np.max(child.stride) > 1 and child.in_channels >= 16): 
    #            setattr(mod, name, BlurPoolConv2d(child))
    #        else: apply_blurpool(child)
    #apply_blurpool(model)
    return model


def test():
    import torch
    from torchsummary import summary
    net = Net(1, 10)
    print(net)
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    print(y.size())
    summary(net, input_size=(3, 32, 32))

    net = Net(4, 126)
    x = torch.randn(2, 3, 96, 96)
    y = net(x)
    print(y.size())
    summary(net, input_size=(3, 96, 96))


if __name__ == "__main__":
    test()

