import torch.nn as nn

__all__ = [
    "ResNet",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "wide_resnet50_2",
    "wide_resnet101_2",
]


def conv3x3(inplanes, outplanes, stride=1, groups=1, padding=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(inplanes,
                     outplanes,
                     kernel_size=3,
                     stride=stride,
                     padding=padding,
                     groups=groups,
                     bias=False)


def conv1x1(inplanes, outplanes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(inplanes,
                     outplanes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                "BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        inchannels,
        frozen_layers=[],
        groups=1,
        width_per_group=64,
        norm_layer=None,
    ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        self.frozen_layers = frozen_layers
        layer_outplanes = [64] + [
            i * block.expansion for i in [64, 128, 256, 512]
        ]
        self.layer_outplanes = list(map(int, layer_outplanes))
        self.layer_strides = [8, 16, 32, 64]

        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(inchannels,
                               self.inplanes,
                               kernel_size=7,
                               stride=2,
                               padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            ))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                ))

        return nn.Sequential(*layers)

    @property
    def layer0(self):
        return nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool)

    def forward(self, x):
        for layer_idx in range(0, 5):
            layer = getattr(self, f"layer{layer_idx}", None)
            if layer is not None:
                x = layer(x)
        return x

    def freeze_layer(self):
        layers = [
            nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool),
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
        ]
        for layer_idx in self.frozen_layers:
            layer = layers[layer_idx]
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        """
        Sets the module in training mode.
        This has any effect only on modules such as Dropout or BatchNorm.

        Returns:
            Module: self
        """
        self.training = mode
        for module in self.children():
            module.train(mode)
        self.freeze_layer()
        return self


def resnet18(**kwargs):
    return build_resnet("resnet18", [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return build_resnet("resnet34", [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return build_resnet("resnet50", [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    return build_resnet("resnet101", [3, 4, 23, 3], **kwargs)


def resnet152(**kwargs):
    return build_resnet("resnet152", [3, 8, 36, 3], **kwargs)


def resnext50_32x4d(**kwargs):
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    return build_resnet("resnext50_32x4d", [3, 4, 6, 3], **kwargs)


def resnext101_32x8d(**kwargs):
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 8
    return build_resnet("resnext101_32x8d", [3, 4, 23, 3], **kwargs)


def wide_resnet50_2(**kwargs):
    kwargs["width_per_group"] = 64 * 2
    return build_resnet("wide_resnet50_2", [3, 4, 6, 3], **kwargs)


def wide_resnet101_2(**kwargs):
    kwargs["width_per_group"] = 64 * 2
    return build_resnet("wide_resnet101_2", [3, 4, 23, 3], **kwargs)


def build_resnet(model_name, layers, **kwargs):
    if model_name in ["resnet18", "resnet34"]:
        model = ResNet(BasicBlock, layers, **kwargs)
    else:
        model = ResNet(Bottleneck, layers, **kwargs)
    return model
