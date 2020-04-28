import torch.nn as nn
import torch.nn.functional as F

class WSDAN(nn.Module):
    def __init__(self, num_class, M=32, net='inception_mixed_6e', pretrained=False):
        super(WSDAN, self).__init__()
        self.num_classes = num_class
        self.M = M
        self.net = net

        if 'inception' in net:
            if net == 'inception_mixed_6e':
                self.features = inception_v3(pretrained=pretrained).get_features_mixed_6e()