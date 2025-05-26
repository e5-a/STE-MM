from .resnet import ResNet, BasicBlock


def Net(w=1, num_classes=10, norm_layer='bn', input_shape=[3, 32, 32], **kwargs):
    dim = (input_shape[1], input_shape[2])
    return ResNet(BasicBlock, [3, 3, 3], w=w, num_classes=num_classes, norm_layer=norm_layer, dim=dim)


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
