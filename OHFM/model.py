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


class MyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(MyLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=num_layers,
                            batch_first=True)
        self.classifer = nn.Linear(hidden_size, output_size)

    def forward(self, input, h, c):
        output, (h, c) = self.lstm(input, (h, c))
        # output : batch, sequence_len, hidden_size*num_directions
        # in training, we only need the last one output[:,-1,:].view([5,1024])\
        output = output[:, -1, :].view([-1, 1024])
        result = self.classifer(output)
        return result, (h, c)


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, sequence_len):
        super(LSTM, self).__init__()

        # self.vgg11 = MyVgg11(output_size)
        self.features = nn.Sequential(*list(torchvision.models.resnet50(pretrained=True).children())[0:-1])  # 2048
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_len = sequence_len

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=num_layers,
                            batch_first=True)

        self.classifier = nn.Linear(hidden_size, output_size)

    def forward(self, input, h, c):
        # input is a 3d tensor (bs, sequence, feature)
        # output is a 3d tensor (bs, sequence, hidden_size)
        output, (h, c) = self.lstm(input, (h, c))

        # result is a 3d tensor (bs, sequence, output_size)
        result = self.classifier(output)
        return result, (h, c)

    def cnn(self, input):
        # input is a 4d tensor (n, c, w, h)
        return self.features(input)

    def frozen(self):
        for param in self.features.parameters():
            param.requires_grad = False

    def parameters(self, recurse=True):
        return filter(lambda p: p.requires_grad, super(LSTM, self).parameters())


class NegativeMapper(nn.Module):
    def __init__(self, bin_feature_len, input_size, nclass):
        super(NegativeMapper, self).__init__()
        self.bin_feature_len = bin_feature_len
        self.input_size = input_size
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=32, num_layers=1, batch_first=True)  # previous:10

        self.mlp1 = nn.Sequential(
            nn.Linear(32, nclass + 1)
        )

        self.mlp3 = nn.Sequential(
            nn.Linear(2048, 32),
            nn.Linear(32, nclass + 1)
        )
        self.mlp4 = nn.Sequential(
            nn.Linear(2048, 1),
            nn.Sigmoid()
        )


    def forward(self, bin_feature, img_features, h, c, alpha_hand):
        bin_feature = bin_feature.view([-1, self.bin_feature_len, self.input_size])
        bin_feature, (_, _) = self.lstm1(bin_feature, (h, c))
        bin_feature = bin_feature[:, -1, :].view([-1, 32])
        bin_result = self.mlp1(bin_feature)

        img_result = self.mlp3(img_features)
        return alpha_hand * bin_result + (1 - alpha_hand) * img_result

