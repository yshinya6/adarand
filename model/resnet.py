import logging
import pdb

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet

logger = logging.getLogger()

# Wrappers of Pytorch official ResNet class
# See https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
    "resnext50_32x4d": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
    "resnext101_32x8d": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
    "wide_resnet50_2": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
    "wide_resnet101_2": "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
}


class BaseResNet(ResNet):
    def __init__(self, num_classes=1000, pretrained=False, finetune=False, modified=False, progress=True, **kwargs):
        super(BaseResNet, self).__init__(block=self.block_class, layers=self.layers, **kwargs)
        if pretrained:
            if isinstance(pretrained, bool):
                logger.info(f"## Load pretrained weight from {model_urls[self.arch]}")
                state_dict = load_state_dict_from_url(model_urls[self.arch], progress=progress)
                self.load_state_dict(state_dict, strict=False)
            elif isinstance(pretrained, str) and not modified:
                logger.info(f"## Load pretrained weight from {pretrained}")
                state_dict = torch.load(pretrained, map_location="cpu")
                self.load_state_dict(state_dict, strict=False)
        self.num_classes = num_classes
        if finetune:
            logger.info(f"## Init. FC layer for {num_classes} classes")
            if self.block_class is BasicBlock:
                self.fc = nn.Linear(512, num_classes)
            else:
                self.fc = nn.Linear(2048, num_classes)
            self.fc.weight.data.normal_(mean=0.0, std=0.01)
            self.fc.bias.data.zero_()
            if modified:
                logger.info(f"## Load pretrained weight from {pretrained}")
                state_dict = torch.load(pretrained, map_location="cpu")
                for k in list(state_dict.keys()):
                    if k.startswith("fc"):
                        del state_dict[k]
                self.load_state_dict(state_dict, strict=False)


class BaseResNetFeature(BaseResNet):
    def __init__(self, *args, **kwargs):
        super(BaseResNetFeature, self).__init__(*args, **kwargs)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        feat = torch.flatten(x, 1)
        pred = self.fc(feat)

        return pred, feat


class CustomResNet(BaseResNet):
    def __init__(self, *args, **kwargs):
        assert "layers" in kwargs
        self.layers = kwargs.pop("layers")
        self.arch = "custom-resnet-" + str(self.layers)
        self.block_class = BasicBlock
        super(CustomResNet, self).__init__(*args, **kwargs)


class ResNet18(BaseResNet):
    def __init__(self, *args, **kwargs):
        self.arch = "resnet18"
        self.block_class = BasicBlock
        self.layers = [2, 2, 2, 2]
        super(ResNet18, self).__init__(*args, **kwargs)


class ResNet18Feat(BaseResNetFeature):
    def __init__(self, *args, **kwargs):
        self.arch = "resnet18"
        self.block_class = BasicBlock
        self.layers = [2, 2, 2, 2]
        super(ResNet18Feat, self).__init__(*args, **kwargs)


class ResNet50(BaseResNet):
    def __init__(self, *args, **kwargs):
        self.arch = "resnet50"
        self.block_class = Bottleneck
        self.layers = [3, 4, 6, 3]
        super(ResNet50, self).__init__(*args, **kwargs)


class ResNet50Feat(BaseResNetFeature):
    def __init__(self, *args, **kwargs):
        self.arch = "resnet50"
        self.block_class = Bottleneck
        self.layers = [3, 4, 6, 3]
        super(ResNet50Feat, self).__init__(*args, **kwargs)


class WRN_50_2(BaseResNet):
    def __init__(self, *args, **kwargs):
        self.arch = "wide_resnet50_2"
        self.block_class = Bottleneck
        self.layers = [3, 4, 6, 3]
        kwargs["width_per_group"] = 64 * 2
        super(WRN_50_2, self).__init__(*args, **kwargs)


class WRN_101_2(BaseResNet):
    def __init__(self, *args, **kwargs):
        self.arch = "wide_resnet101_2"
        self.block_class = Bottleneck
        self.layers = [3, 4, 23, 3]
        kwargs["width_per_group"] = 64 * 2
        super(WRN_101_2, self).__init__(*args, **kwargs)
