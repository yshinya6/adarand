import logging
import pdb

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.models.vision_transformer import VisionTransformer

logger = logging.getLogger()

# Wrappers of Pytorch official ResNet class
# See https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

model_urls = {
    "vit-b-32": "https://download.pytorch.org/models/vit_b_32-d86f8d99.pth",
    "vit-b-16": "https://download.pytorch.org/models/vit_b_16-c867db91.pth",
}


class BaseViT(VisionTransformer):
    def __init__(self, num_classes=1000, pretrained=False, finetune=False, modified=False, progress=True, **kwargs):
        super(BaseViT, self).__init__(**kwargs)
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
            self.fc = nn.Linear(self.hidden_dim, num_classes)
            self.fc.weight.data.normal_(mean=0.0, std=0.01)
            self.fc.bias.data.zero_()
            if modified:
                logger.info(f"## Load pretrained weight from {pretrained}")
                state_dict = torch.load(pretrained, map_location="cpu")
                for k in list(state_dict.keys()):
                    if k.startswith("fc"):
                        del state_dict[k]
                self.load_state_dict(state_dict, strict=False)


class BaseViTFeature(BaseViT):
    def __init__(self, *args, **kwargs):
        super(BaseViTFeature, self).__init__(*args, **kwargs)

    def forward(self, x):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        feat = torch.flatten(x, 1)
        pred = self.fc(feat)

        return pred, feat


class ViT_B_32_Feat(BaseViTFeature):
    def __init__(self, *args, **kwargs):
        self.arch = "vit-b-32"
        super(ViT_B_32_Feat, self).__init__(
            image_size=224,
            patch_size=32,
            num_layers=12,
            num_heads=12,
            hidden_dim=768,
            mlp_dim=3072,
            progress=True,
            **kwargs
        )


class ViT_B_16_Feat(BaseViTFeature):
    def __init__(self, *args, **kwargs):
        self.arch = "vit-b-16"
        super(ViT_B_16_Feat, self).__init__(
            image_size=224,
            patch_size=16,
            num_layers=12,
            num_heads=12,
            hidden_dim=768,
            mlp_dim=3072,
            progress=True,
            **kwargs
        )


class ViT_S_16_Feat(BaseViTFeature):
    def __init__(self, *args, **kwargs):
        self.arch = "vit-s-16"
        super(ViT_S_16_Feat, self).__init__(
            image_size=224,
            patch_size=16,
            num_layers=12,
            num_heads=6,
            hidden_dim=384,
            mlp_dim=1536,
            progress=True,
            **kwargs
        )
