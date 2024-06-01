import torch
import torchvision as tv
import numpy as np
import normflows as nf
import torch.nn as nn
import argparse
from tqdm import tqdm
from tensorboardX import SummaryWriter
from diffusers import DDPMScheduler, UNet2DModel

class gender_resnet(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        in_channels, out_channels = 3, 64
        self.fc = nn.Linear(1000, 2)
        self.softmax = nn.LogSoftmax()
    def forward(self, x, need_log=True):
        if(not need_log):
            return torch.exp(self.softmax(self.fc(self.model(x))))
        return self.softmax(self.fc(self.model(x)))
    

def get_resnet_through_name(name, pretrained=True):
    if name == 'resnet18':
        model = tv.models.resnet18(pretrained=pretrained)
    elif name == 'resnet34':
        model = tv.models.resnet34(pretrained=pretrained)
    elif name == 'resnet50':
        model = tv.models.resnet50(pretrained=pretrained)
    elif name == 'resnet101':
        model = tv.models.resnet101(pretrained=pretrained)
    elif name == 'resnet152':
        model = tv.models.resnet152(pretrained=pretrained)
    else:
        assert(0)
    return gender_resnet(model)



def UNETModel():
    return UNet2DModel(
    sample_size=112,           # the target image resolution
    in_channels=3,            # the number of input channels, 3 for RGB images
    out_channels=3,           # the number of output channels
    layers_per_block=1,       # how many ResNet layers to use per UNet block
    block_out_channels=(32, 32, 64), # Roughly matching our basic unet example
    down_block_types=( 
        "DownBlock2D",        # a regular ResNet downsampling block
        "AttnDownBlock2D",    # a ResNet downsampling block with spatial self-attention
        "AttnDownBlock2D",
    ), 
    up_block_types=(
        "AttnUpBlock2D", 
        "AttnUpBlock2D",      # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",          # a regular ResNet upsampling block
      ),
)
