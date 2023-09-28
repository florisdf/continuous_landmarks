from typing import Literal

from torch import nn
from torchvision.models.mobilenetv3 import mobilenet_v3_large

from .convnext import convnext_small


class FeatureExtractor(nn.Module):
    def __init__(self, model_name: Literal['ConvNeXt', 'MobileNetV3']):
        super().__init__()
        if model_name == 'ConvNeXt':
            model = convnext_small()
            self.feature_size = model.head.in_features
            model.norm = nn.Identity()
            model.head = nn.Identity()
        elif model_name == 'MobileNetV3':
            model = mobilenet_v3_large()
            self.feature_size = model.classifier[0].in_features
            model.classifier = nn.Identity()
        else:
            raise ValueError(f'Unknown model name "{model_name}"')

        self.model = model

    def forward(self,  x):
        return self.model(x)
