# TSM_TDN Combined Model (Version 1)

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.init import normal_, constant_
import torch.nn.functional as F
from TDMops.base_module import *
from TDMops.tdn_net import TDN_Net
from TSMops.models import TSN

class TSM_TDM(nn.Module):

    def __init__(self,resnet_model,resnet_model1,apha,belta):
        super(TSM_TDM, self).__init__()

        self.conv1 = list(resnet_model.children())[0]
        self.bn1 = list(resnet_model.children())[1]
        self.relu = nn.ReLU(inplace=True)

        #Temporal Shift Model
        self.tsm = TSN()

        #Temporal Difference Model
        self.tdm = TDN_Net()

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.fc = list(resnet_model.children())[8]
        self.avgpool = nn.AvgPool2d(7, stride=1)


    def forward(self, x):
        
        x = self.tsm(x)

        x = self.tdm(x)

        x = self.maxpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def tsm_tdm(base_model=None,num_segments=8,pretrained=True, **kwargs):
    if("50" in base_model):
        resnet_model = fbresnet50(num_segments, pretrained)
        resnet_model1 = fbresnet50(num_segments, pretrained)
    else:
        resnet_model = fbresnet101(num_segments, pretrained)
        resnet_model1 = fbresnet101(num_segments, pretrained)

    if(num_segments is 8):
        model = TSM_TDM(resnet_model,resnet_model1,apha=0.5,belta=0.5)
    else:
        model = TSM_TDM(resnet_model,resnet_model1,apha=0.75,belta=0.25)
    return model