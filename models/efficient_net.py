import math
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

def efficientnet_b1(pretrained=False, num_classes=1000):
    model = EfficientNet.from_pretrained('efficientnet-b1', num_classes=1000)
    return model