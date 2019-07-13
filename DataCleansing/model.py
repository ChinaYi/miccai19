'''
Created on Sep 29, 2018

@author: yi
'''
from __future__ import print_function
from __future__ import division
import cfg
import math
import torch
import torch.nn as nn
from torch.autograd import Function
import torchvision
import torch.nn.init as init
from torch import nn,optim, LongTensor
import torch.nn.functional as F
from torch.utils import model_zoo

import numpy as np





        
class MyResnet(nn.Module):
    def __init__(self, classes):
        super(MyResnet, self).__init__()
        resnet50 = torchvision.models.resnet50(pretrained = True)
        self.features = nn.Sequential(*list(resnet50.children())[0:-1]) #2048
        self.fc = nn.Linear(2048, classes, bias = True)

    def forward(self, x):
        feature = self.features(x).view(-1, 2048)
        result = self.fc(feature)
        return feature, result
    
