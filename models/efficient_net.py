import math
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet


def efficientnet_b1():
    model = EfficientNet.from_pretrained('efficientnet-b1')
    return model.cuda()


def efficientnet_b2():
    model = EfficientNet.from_pretrained('efficientnet-b2')
    return model


def efficientnet_b3():
    model = EfficientNet.from_pretrained('efficientnet-b3')
    return model


def efficientnet_b4():
    model = EfficientNet.from_pretrained('efficientnet-b4')
    return model


def efficientnet_b5():
    model = EfficientNet.from_pretrained('efficientnet-b5').extract_features
    return model


def efficientnet_b6():
    model = EfficientNet.from_pretrained('efficientnet-b6')
    return model


def efficientnet_b7():
    model = EfficientNet.from_pretrained('efficientnet-b7')
    return model