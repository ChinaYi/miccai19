# Be sure that both resnet50 and Hard Negative mapper are trained!
from __future__ import print_function
from __future__ import division

import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils import data
import numpy as np

from dataset import *
# utf-8 encoding

from functools import partial

try:
    import cPickle as pickle  # python 2
except ImportError as e:
    import pickle  # python 3
pickle.load = partial(pickle.load, encoding="latin1")
pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
# --------------

from cfg import *
import utils
from model import *
from dataset import *

CUDA = True if torch.cuda.is_available() else False


def cal_avgres(result):
    res_dict = {}
    if len(result) == 0:
        return 0, 1.
    for predict in result:
        if predict not in res_dict.keys():
            res_dict[predict] = 1
        else:
            res_dict[predict] += 1

    max_rate = 0
    max_predict = 0
    for key, value in res_dict.items():
        if value / len(result) > max_rate:
            max_rate = value / len(result)
            max_predict = key
    return max_predict, max_rate


def mapper(model, result_record, neg_index, img_feature, bin_num, interval, bound):
    for local_num in range(interval, bound, interval):
        mode = utils.get_mode(result_record[max(0, len(result_record) - local_num):len(result_record):])[0]
        if mode != neg_index:
            return mode, 1.0

    bin_feature = utils.mapper_feature(l=result_record, global_num= bin_num, neg = neg_index)
    bin_feature = torch.from_numpy(bin_feature).view([-1, bin_num]).cuda()

    process_feature = torch.from_numpy(np.array([len(result_record) * 1.0])).float().view([1, 1]).cuda()
    img_feature = torch.from_numpy(img_feature).view([1, 2048]).cuda()
    hx, cx = torch.zeros((1, 1, 32)).cuda(), torch.zeros((1, 1, 32)).cuda()

    res = model(bin_feature, img_feature, hx, cx, alpha_hand = 0.95)
    res = F.softmax(res, dim=1)

    predict = int(res.max(1)[1][0])
    c = float(res.max(1)[0][0])
    return predict, c


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--savepath', type = str)
    parser.add_argument('--enable_pki', type = bool)
    args = parser.parse_args()

    hn_mapper = torch.load('models/{}-mapper.pth'.format(args.dataset), pickle_module=pickle)
    resnet50 = torch.load('models/{}-resnet50.pth'.format(args.dataset), pickle_module=pickle)
    # hypers = pickle.load('hypers_pki_{}.pkl'.format(args.dataset))
    print(hn_mapper)
    hn_mapper.eval()
    resnet50.eval()

    feature_folder = 'image_folder'
    negative_index = 8 if args.dataset == 'm2cai16-workflow-5' else 7
    result_path = 'test_result/{}'.format(args.dataset)

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    if CUDA:
        hn_mapper = hn_mapper.cuda()
        resnet50 = resnet50.cuda()

    if args.dataset == 'm2cai16-workflow-5':
        interval = 5
        bound = 65
    else:
        interval = 10
        bound = 100

    transform = utils.transform_factory(split='test', dataset=args.dataset)
    test_dataset = VideoDataset_(root = os.path.join(args.dataset, 'test_dataset'),
                                dataset=args.dataset,
                                special_list = [], filter_type = 'not_in',
                                feature_folder = feature_folder, ground_truth_folder = 'annotation_folder')

    for video_index in range(test_dataset.__len__()):
        video_imgs, video_labels, video_name = test_dataset.__getitem__(video_index, video_index + 1)

        video_dataset = VideoDataset(video_imgs, video_labels, transform)
        video_loader = data.DataLoader(dataset=video_dataset, shuffle=False, batch_size=1, drop_last=False) # tackle one by one

        video_name = video_name[0]
        predict_initial_record = []  # record the predict result before mapping
        predict_pki_record = []  # record the predict result after pki

        lines = [i for i in range(len(video_labels[0]) + 1)]
        lines[0] = 'Frame\tPhase\n'

        # parameter in PKI
        implict_pn = 0

        transfer_count_dict = {k: 1 for k in range(negative_index)}
        pre_dict = {k: 0 for k in range(negative_index)}

        with torch.no_grad():
            for iter, (imgs, labels, img_names) in enumerate(video_loader):
                if CUDA:
                    imgs, labels = imgs.cuda(), labels.cuda()
                bs, ncrops, c, h, w = imgs.size()
                feature, res = resnet50(imgs.view(-1, c, h, w))

                avg_res = res.view(bs, ncrops, -1).mean(1)
                avg_res = F.softmax(avg_res, dim=1)

                predict = int(avg_res.max(1)[1][0])
                confidence = float(avg_res.max(1)[0][0])

                predict_initial_record.append(predict + 1)  # 0 is padding

                if predict == negative_index:
                    img_feature = np.load(
                        '{}/test_dataset/feature_folder/{}/{}.npy'.format(args.dataset, video_name, iter))
                    predict = mapper(hn_mapper, predict_initial_record, negative_index + 1, img_feature, 240, interval, bound)[0] - 1

                lines[iter + 1] = '{}\t{}\n'.format(iter, predict)

            print('testing {}-{} done!'.format(video_name, video_index))
            lines = utils.expand(lines, dataset=args.dataset, timeF=5)[
                    0:config[args.dataset]['testset_len'][video_name]]
            with open(os.path.join(result_path, video_name + '_pred.txt'), 'w') as f:
                for line in lines:
                    f.write(str(line))
