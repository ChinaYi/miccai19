'''
    Train the resnet model w/wo data cleansing strategy.
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

import datetime
import os
try:
    import cPickle as pickle    #python 2
except ImportError as e:
    import pickle   #python 3

CUDA = True if torch.cuda.is_available() else False
l2_decay = 5e-5
epochs = 5
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
    best_perform = 0.
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

        if test_acc > best_perform:
            best_perform = test_acc
            torch.save(model, '{}/best.pth'.format(savepath))

        print('Iter-{}: Training/Testing Acc: {}/{}, Training/Testing Loss: {}/{}'.format(epoch, train_acc, test_acc, train_loss_item, test_loss_item))

        logger.scalar_summary('test_accuracy', test_acc, epoch)
        logger.scalar_summary('test_loss', test_loss_item, epoch)
        logger.scalar_summary('train_loss', train_loss_item, epoch)
        logger.scalar_summary('train_accuracy', train_acc, epoch)

        if epoch % save_interval == 0:
            torch.save(model, '{}/_{}.pth'.format(save_path, epoch))

    print('Training done!')


def get_resnet_result(model, dataset, result_save_folder):
    if not os.path.exists(result_save_folder):
        os.makedirs(result_save_folder)
    model.eval()
    if CUDA:
        model.cuda()
    transform = utils.transform_factory(split='test', dataset = dataset)
    cleansing_video_dataset = VideoDataset_(root=os.path.join(args.dataset, 'test_dataset'), dataset=dataset,
                                            special_list=[], filter_type='not_in',
                                            feature_folder=feature_floder,
                                            ground_truth_folder='annotation_folder')
    for video_index in range(cleansing_video_dataset.__len__()):
        video_imgs, video_labels, video_name = cleansing_video_dataset.__getitem__(video_index, video_index + 1)

        video_dataset = VideoDataset(video_imgs, video_labels, transform)
        video_loader = data.DataLoader(dataset=video_dataset, shuffle=False, batch_size=1,
                                           drop_last=False)  # tackle one by one

        video_name = video_name[0]
        lines = [i for i in range(len(video_labels[0]) + 1)]
        lines[0] = 'Frame\tPhase\n'
        result_save_path = os.path.join(result_save_folder, video_name + '_pred.txt')
        with torch.no_grad():
            for iter, (imgs, labels, img_names) in enumerate(video_loader):
                if CUDA:
                    imgs, labels = imgs.cuda(), labels.cuda()
                bs, ncrops, c, h, w = imgs.size()
                feature, res = model(imgs.view(-1, c, h, w))

                avg_res = res.view(bs, ncrops, -1).mean(1)
                avg_res = F.softmax(avg_res, dim=1)

                predict = avg_res.max(1)[1].item()
                lines[iter + 1] = '{}\t{}\n'.format(iter, predict)
                
            print('testing video {} done!'.format(video_name,))
            lines = utils.expand(lines, dataset=dataset, timeF=5)[
                    0:config[dataset]['testset_len'][video_name]]
            with open(result_save_path, 'w') as f:
                for line in lines:
                    f.write(str(line))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, choices=['m2cai16-workflow-5', 'cholec80-workflow-5'],
                        help='choose the dataset')
    parser.add_argument('--savepath', type=str, help='path to save the cleansing results')  #
    parser.add_argument('--ground_truth_path', type=str, help='path of the ground truth files')  # which gt file we use
    parser.add_argument('--load', type=str, default='no',
                        help='path for the trained model, default is no')  # The path of trained model
    parser.add_argument('--logdir', type=str, default='./logs', help='tensorboard logs dir')
    parser.add_argument('--repeat_times', type=int, default=1, help='repeat the experiment to eliminate randomness')
    parser.add_argument('--use_pymatbridge', type=bool, default=False, help='execute the matlab .m file within the code')
    args = parser.parse_args()

    train_transform = utils.transform_factory(split='train', dataset=args.dataset)
    test_transform = utils.transform_factory(split='test', dataset=args.dataset)

    savepath = args.savepath

    if not os.path.exists(savepath):
        os.makedirs(savepath)

    starttime = datetime.datetime.now()
    print('code starts at ', starttime)

    batch_size = 64
    epochs = 8
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
            dataset=CleansingDataset(root=os.path.join(args.dataset, 'train_dataset'), transform=train_transform,
                                     dataset=args.dataset,
                                     special_list=[], filter_type='not_in',
                                     feature_folder=feature_floder, ground_truth_folder=args.ground_truth_path),
            batch_size=batch_size, shuffle=True, drop_last=True
        )
        test_loader = data.DataLoader(
            dataset=CleansingDataset(root=os.path.join(args.dataset, 'test_dataset'), transform=test_transform,
                                     dataset=args.dataset,
                                     special_list=[], filter_type='not_in',
                                     feature_folder=feature_floder, ground_truth_folder='annotation_folder'),
            batch_size=batch_size, shuffle=True, drop_last=True
        )

        train(model, train_loader, test_loader, epochs, args.logdir, savepath)

    get_resnet_result(model, args.dataset, savepath)
    endtime = datetime.datetime.now()
    print('code ends at ', endtime - starttime)

    if args.use_pymatbridge:
        from pymatbridge import Matlab
        mlab = Matlab()
        mlab.start()
        results = mlab.run_code('run {}/eva/Main.m'.format(args.dataset))
        meanJacc, stdJacc, meanAcc, stdAcc = mlab.get_variable('meanJacc'), mlab.get_variable(
            'stdJacc'), mlab.get_variable('meanAcc'), mlab.get_variable('stdAcc')
