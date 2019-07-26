'''
Created on Sep 30, 2018

@author: yi
'''

from __future__ import print_function
from __future__ import division

import argparse
import math
import random
import numpy as np
import time
import datetime

try:
    import cPickle as pickle  # python 2
except ImportError as e:
    import pickle  # python 3

from cfg import *
import utils
from model import *
from dataset import *

import torch
from torch.autograd import Variable
from torch.utils import data
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets.folder import default_loader


batch_size = 64
epochs = 5
lr = 5e-4
l2_decay = 5e-5



def test(model, test_loader, alpha):
    model.eval()
    loss_layer = nn.CrossEntropyLoss()

    accuracy_count = 0
    total_count = 0
    loss_count = 0
    hx, cx = torch.zeros((1, batch_size, 32)).cuda(), torch.zeros((1, batch_size, 32)).cuda()

    with torch.no_grad():
        for iter, (bin_feature, img_features, label) in enumerate(test_loader):
            (bin_feature, img_features, label) \
                = bin_feature.cuda(), img_features.cuda(), label.cuda()

            res = model(bin_feature, img_features, hx, cx, alpha)

            res = F.softmax(res, dim=1)
            accuracy_count += res.max(1)[1].eq(label).sum().item()
            total_count += len(label)
            loss_count += loss_layer(res, label).item()

    return accuracy_count / total_count, loss_count / total_count



def pretrain(dataset, input_size, savepath):
    import os
    savepath = os.path.join(savepath, dataset)
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    if dataset == 'm2cai16-workflow-5':
        train_pair = ['workflow_video_{:>02d}'.format(i) for i in range(1, 28)]
        test_pair = ['test_workflow_video_{:>02d}'.format(i) for i in range(1, 15)]
        nclass = 8
    else:
        train_pair = ['video{:>02d}'.format(i) for i in range(1, 41)]
        test_pair = ['video{:>02d}'.format(i) for i in range(41, 81)]
        nclass = 7

    mapper_train_set = MapperDataset(dataset_path='{}/train_dataset'.format(dataset), filter=train_pair,
                                     filter_type='in', prefix='')
    mapper_test_set = MapperDataset(dataset_path='{}/test_dataset'.format(dataset), filter=test_pair, filter_type='in',
                                    prefix='')

    mapper_train_loader = data.DataLoader(dataset=mapper_train_set, shuffle=True, batch_size=batch_size, drop_last=True)
    mapper_test_loader = data.DataLoader(dataset=mapper_test_set, shuffle=True, batch_size=batch_size, drop_last=True)

    for alpha in [0.95]:
        model = NegativeMapper(bin_feature_len=240, input_size= input_size, nclass=nclass)
        print('Training start!')
        loss_layer = nn.CrossEntropyLoss()

        model = model.cuda()

        best_perform = 0.
        for epoch in range(1, epochs):
            model.train()
            LEARNING_RATE = lr / math.pow((1 + 10 * (epoch - 1) / epochs), 0.75)
            optimizer = torch.optim.Adam([
                {'params': model.parameters()},
            ], lr=LEARNING_RATE * 10, weight_decay=l2_decay)

            train_accuracy_count = 0
            train_total_count = 0
            train_loss_count = 0
            hx, cx = torch.zeros((1, batch_size, 32)).cuda(), torch.zeros((1, batch_size, 32)).cuda()
            for iter, (bin_feature, img_features, label) in enumerate(mapper_train_loader):
                (bin_feature, img_features, label) = bin_feature.cuda(), img_features.cuda(), label.cuda()

                res = model(bin_feature, img_features, hx, cx, alpha)

                loss = loss_layer(res, label)

                train_loss_count += loss.item()
                train_total_count += len(label)
                train_accuracy_count += res.max(1)[1].eq(label).sum().item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_accuracy_count = train_accuracy_count / train_total_count
            train_loss_count = train_loss_count / train_total_count

            test_acc, test_loss = test(model, mapper_test_loader, alpha)
            if test_acc > best_perform:
                best_perform = test_acc
                torch.save(model, '{}/best.pth'.format(savepath,))
            print('epoch {}, Train/Test Loss: {}/{}, Train/Test Acc: {}/{}'.format(epoch, train_loss_count, test_loss,
                                                           train_accuracy_count, test_acc))

            torch.save(model, '{}/{}.pth'.format(savepath, epoch));


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str)
    args = parser.parse_args()

    if args.dataset == 'm2cai16-workflow-5':
        input_size = 18
    else:
        input_size = 16
    pretrain(dataset=args.dataset, input_size=input_size, savepath='hmapper')
