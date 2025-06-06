import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, norm_layer='bn', dim=(32, 32)):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        if norm_layer == 'bn':
            self.bn1 = nn.BatchNorm2d(planes)
        else:
            self.bn1 = nn.LayerNorm([planes, *dim])
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        if norm_layer == 'bn':
            self.bn2 = nn.BatchNorm2d(planes)
        else:
            self.bn2 = nn.LayerNorm([planes, *dim])
        self.relu2 = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(planes) if norm_layer == 'bn' else nn.LayerNorm([planes, *dim])
            )

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu2(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, w=1, num_classes=10, norm_layer='bn', dim=(32, 32)):
        super(ResNet, self).__init__()
        self.in_planes = w * 16

        self.conv1 = nn.Conv2d(3, w * 16, kernel_size=3, stride=1, padding=1, bias=False)
        if norm_layer == 'bn':
            self.bn1 = nn.BatchNorm2d(w * 16)
        else:
            self.bn1 = nn.LayerNorm([w * 16, *dim])
        self.relu1 = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, w * 16, num_blocks[0], stride=1, norm_layer=norm_layer, dim=(dim[0], dim[1]))
        self.layer2 = self._make_layer(block, w * 32, num_blocks[1], stride=2, norm_layer=norm_layer, dim=(dim[0]//2, dim[1]//2))
        self.layer3 = self._make_layer(block, w * 64, num_blocks[2], stride=2, norm_layer=norm_layer, dim=(dim[0]//4, dim[1]//4))
        self.linear = nn.Linear(w * 64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride, norm_layer, dim):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, norm_layer, dim))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

