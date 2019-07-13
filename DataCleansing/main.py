'''
Created on Sep 29, 2018

@author: yi
'''
from __future__ import print_function
from __future__ import division

import argparse
import utils

from model import *
from dataset import *
from cfg import *
from logger import Logger

import torch
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F

import math
import datetime
import os
try:
    import cPickle as pickle    #python 2
except ImportError as e:
    import pickle   #python 3



CUDA = True if torch.cuda.is_available() else False

l2_decay = 5e-5
epochs = 10
lr = 5e-4
momentum = 0.9
save_interval = 1
batch_size = 64


def test(model, test_loader):
    loss_layer = nn.CrossEntropyLoss()
    model.eval()

    acc = 0.
    loss_item = 0.
    num_of_images = 0
    with torch.no_grad():
        for iter, (imgs, labels, img_names) in enumerate(test_loader):
            if CUDA:
                imgs, labels = imgs.cuda(), labels.cuda()

            bs, ncrops, c, h, w = imgs.size()
            feature, res = model(imgs.view(-1, c, h, w))

            avg_res = res.view(bs, ncrops, -1).mean(1)

            acc += avg_res.max(1)[1].eq(labels).sum().item()
            loss_item += loss_layer(avg_res, labels).item()
            num_of_images += bs

    return acc / num_of_images, loss_item / num_of_images


def train(model, train_loader, test_loader, epochs, logdir, save_path):
    logger = Logger(logdir)
    loss_layer = nn.CrossEntropyLoss()

    best_model_acc = 0.
    for epoch in range(1, epochs):
        model.train()
        LEARNING_RATE = lr / math.pow((1 + 10 * (epoch - 1) / epochs), 0.75)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=l2_decay)

        train_acc = 0
        train_loss_item = 0.
        train_num_of_images = 0

        for iter, (imgs, labels, img_names) in enumerate(train_loader):

            if CUDA:
                imgs, labels = imgs.cuda(), labels.cuda()

            feature, res = model(imgs)
            loss = loss_layer(res, labels)

            train_acc += res.max(1)[1].eq(labels.data).sum().item()
            train_loss_item += loss.item()
            train_num_of_images += len(imgs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        train_acc = train_acc / train_num_of_images
        train_loss_item = train_loss_item / train_num_of_images
        test_acc, test_loss_item = test(model, test_loader)
        if test_acc > best_model_acc:
            best_model_acc = test_acc
            torch.save(torch.save(model, '{}/_best.pth'.format(save_path)))
        print('Iter-{}: Training/Testing Acc: {}/{}, Training/Testing Loss: {}/{}'.format(epoch, train_acc, test_acc, train_loss_item, test_loss_item))

        logger.scalar_summary('test_accuracy', test_acc, epoch)
        logger.scalar_summary('test_loss', test_loss_item, epoch)
        logger.scalar_summary('train_loss', train_loss_item, epoch)
        logger.scalar_summary('train_accuracy', train_acc, epoch)

    print('Training done!')

def list_indicator(target, indicator):
    # Naive approach
    out = []
    for i in range(len(target)):
        if indicator[i] > 0:
            out.append(target[i])
    return out

def cleansing(model, dataset, test_dataset, negative_index, cleansing_res_file_path):
    model.eval()
    if CUDA:
        model.cuda()

    transform = utils.transform_factory(split='test', dataset=dataset)
    loss_layer = nn.CrossEntropyLoss()

    for video_index in range(test_dataset.__len__()):
        video_imgs, video_labels, video_name = test_dataset.__getitem__(video_index, video_index + 1)

        video_dataset = VideoDataset(video_imgs, video_labels, transform)
        video_loader = data.DataLoader(dataset=video_dataset, shuffle=False, batch_size=1, drop_last=False) # tackle one by one

        video_name = video_name[0]
        with torch.no_grad():
            positive_samples = []
            negative_samples = []

            negative_count = 0
            positive_count = 0

            video_len = 0
            correct_count = 0

            for iter, (imgs, labels, img_names) in enumerate(video_loader):
                video_len += len(labels)

                if CUDA:
                    imgs, labels = imgs.cuda(), labels.cuda()
                bs, ncrops, c, h, w = imgs.size()
                feature, res = model(imgs.view(-1, c, h, w))

                avg_res = res.view(bs, ncrops, -1).mean(1)
                avg_res = F.softmax(avg_res, dim=1)

                predict = avg_res.max(1)[1]
                confidence = avg_res.max(1)[0]

                correct_count += predict.eq(labels).sum().item()

                negative_indicators = (torch.ones_like(labels).byte() ^ predict.eq(labels)).cpu()
                positive_indicators = (torch.ones_like(labels).cpu().byte() ^ negative_indicators).numpy()
                negative_indicators = negative_indicators.numpy()

                positive_samples_names = [img_name.split('/')[-1].split('.')[0] for img_name in
                                          list_indicator(img_names, positive_indicators)]
                positive_samples_labels = list_indicator(labels.cpu().numpy(), positive_indicators)

                positive_samples += list(zip(positive_samples_names, positive_samples_labels))
                positive_count += len(positive_samples_names)

                # negative_samples name and labels
                negative_samples_names = [img_name.split('/')[-1].split('.')[0] for img_name in
                                          list_indicator(img_names, negative_indicators)]
                negative_samples_labels = [negative_index] * len(negative_samples_names)  ## n + 1 classification tasks
                negative_samples_confidences = list_indicator(confidence, negative_indicators)
                negative_samples_original_labels = list_indicator(labels.cpu().numpy(), negative_indicators)

                negative_samples += list(
                    zip(negative_samples_names, negative_samples_labels))
                negative_count += len(negative_samples_names)

            samples = positive_samples + negative_samples
            samples = sorted(samples, key=lambda a: int(a[0]))

            print('Writing cleansing results to save folder....')

            with open(os.path.join(cleansing_res_file_path, '{}.txt'.format(video_name)), 'w') as f:
                for (name, label) in samples:
                    f.write('{}\t{}\n'.format(name, config[dataset]['reverse_mapping_dict'][int(label)]))

            print('Video {}: positive samples {}, negative samples {}, correct is {}'.format(video_name,
                                                                                             positive_count / video_len,
                                                                                             negative_count / video_len,
                                                                                             correct_count / video_len))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, choices = ['m2cai16-workflow-5', 'cholec80-workflow-5'], help = 'choose the dataset')
    parser.add_argument('--cleansing', type=str, help = 'videos to perform cleasning, seperated by #')
    parser.add_argument('--savepath', type=str, help = 'path to save the cleansing results')  #
    parser.add_argument('--ground_truth_path', type=str, help = 'path of the ground truth files')  # which gt file we use
    parser.add_argument('--load', type=str, default = 'no', help = 'path for the trained model, default is no')  # The path of trained model
    parser.add_argument('--logdir', type=str, default='./logs', help = 'tensorboard logs dir')
    args = parser.parse_args()

    train_transform = utils.transform_factory(split='train', dataset=args.dataset)
    test_transform = utils.transform_factory(split='test', dataset=args.dataset)

    videos_for_cleansing = args.cleansing  # videos are seperated with #, i.e. 'workflow_video_01#workflow_video_02'
    savepath = os.path.join(args.savepath, videos_for_cleansing)

    if not os.path.exists(savepath):
        os.makedirs(savepath)

    starttime = datetime.datetime.now()
    print('code starts at ', starttime)

    batch_size = 64
    epochs = 9
    feature_floder = 'image_folder'
    negative_index = 8 if args.dataset == 'm2cai16-workflow-5' else 7
    model = MyResnet(classes=negative_index)

    if args.load != 'no':
        model = utils.load_model(model, args.load)

    if CUDA:
        model = model.cuda()

    print('Detail : batch_size : {}, epochs: {}, feature floder :{}'.format(batch_size, epochs, feature_floder))

    if args.load == 'no':
        train_loader = data.DataLoader(
            dataset= CleansingDataset(root = os.path.join(args.dataset, 'train_dataset'), transform=train_transform,
                                      dataset=args.dataset,
                                      special_list=videos_for_cleansing.split('#'), filter_type='not_in',
                                      feature_folder = feature_floder, ground_truth_folder = args. ground_truth_path),
            batch_size=batch_size, shuffle= True, drop_last=True
        )
        test_loader = data.DataLoader(
            dataset= CleansingDataset(root=os.path.join(args.dataset, 'train_dataset'), transform=test_transform,
                                     dataset=args.dataset,
                                     special_list=videos_for_cleansing.split('#'), filter_type='in',
                                     feature_folder=feature_floder, ground_truth_folder=args.ground_truth_path),
            batch_size=batch_size, shuffle=True, drop_last=True
        )

        train(model, train_loader, test_loader, epochs, args.logdir, savepath)

    if videos_for_cleansing == 'no':
        #         test_result_resnet(model, dataset = args.dataset, expanding_timeF = 5, negative_index = negative_index, result_path = '{}_test_result'.format(args.dataset))
        test_dataset_path = 'cholec80-workflow-5/test_dataset'
        get_feature_resnet(model, dataset=args.dataset, test_dataset_path=test_dataset_path, expanding_timeF=5,
                           negative_index=negative_index, special_list=[], filter_type='not_in')
    #
    else:
        cleansing_video_dataset = VideoDataset_(root = os.path.join(args.dataset, 'train_dataset'),
                                               dataset=args.dataset,
                                               special_list = videos_for_cleansing.split('#'), filter_type = 'in',
                                               feature_folder = feature_floder, ground_truth_folder = args. ground_truth_path)

        cleansing(model, args.dataset, cleansing_video_dataset, negative_index, savepath)
        # test_dataset_path = 'cholec80-workflow-5/train_dataset'
        # get_feature_resnet(model, dataset=args.dataset, test_dataset_path=test_dataset_path, expanding_timeF=5,
        #                    negative_index=negative_index, special_list=videos_separate_with_shop.split('#'),
        #                    filter_type='in')

    endtime = datetime.datetime.now()
    print('code ends at ', endtime - starttime)