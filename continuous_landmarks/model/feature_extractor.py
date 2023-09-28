from torch import nn
from torchvision.models.mobilenetv3 import mobilenet_v3_large

from continuous_landmarks.model.convnext import convnext_small


class FeatureExtractor(nn.Module):
    def __init__(self, model_name: str):
        if model_name == 'ConvNeXt':
            model = convnext_small()
            model.norm = nn.Identity()
            model.head = nn.Identity()
        elif model_name == 'MobileNetV3':
            model = mobilenet_v3_large()
            model.classifier = nn.Identity()
        else:
            raise ValueError(f'Unknown model name "{model_name}"')

        self.model = model

    def forward(self,  x):
        return self.model(x)
