import numpy as np
import torch.nn as nn
from torchvision import transforms
from robomimic.models.base_nets import Module

# Wrap R3M into a robomimic's module.
# NOTE: This is a hack to use R3M in this codebase as an observation encoder,
# with code partly borrowed from robomimic.models.base_nets.
# Redo this more elegantly.
class R3M_Module(Module):
    def __init__(self, R3M_obj):
        super().__init__()
        self.R3M_obj = R3M_obj
        self.bn = nn.BatchNorm1d(self.R3M_obj.outdim)

    def forward(self, x, **kwargs):
        # "Unprocess" images so that they are in [0, 255] and upsample them to 224x224.
        x *= 255
        x = x.int()
        if (x.shape[-1] != 224 or x.shape[-2] != 224):
            preprocess = nn.Sequential(
                transforms.Resize(224)
            )
            x = preprocess(x)
        x = self.R3M_obj.forward(x, **kwargs)
        x = self.bn(x)
        return x

    def output_shape(self, input_shape=None):
        # The return dim of a BN layer is the same is its input dim (R3M's output dim)
        return [self.R3M_obj.outdim]
