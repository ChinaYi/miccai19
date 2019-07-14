from cfg import *

import torch
from torchvision.transforms import *
import random
import numpy as np
import os

def expand(lines, dataset, timeF=5):
    new_lines = []
    new_lines.append('Frame\tPhase\n')

    new_index = 0
    for line in lines[1:]:
        phase = line.split('\t')[1].strip()
        if phase.isdigit():
            phase = int(phase)
        else:
            phase = float(phase)
        for i in range(timeF):
            if isinstance(phase, int):
                new_lines.append(
                    '{}\t{}\n'.format(new_index, config[dataset]['reverse_mapping_dict'][phase]))  # 0\t Preparation
            else:
                new_lines.append('{}\t{}\n'.format(new_index, phase))
            new_index += 1
    return new_lines


def get_mode(arr):
    if len(arr) == 0:
        return 1, 1

    max_p = 1
    max_c = 0
    arr_appear = {}
    for phase in arr:
        if phase not in arr_appear.keys():
            arr_appear[phase] = 1
            if arr_appear[phase] > max_c:
                max_c = arr_appear[phase]
                max_p = phase
        else:
            arr_appear[phase] += 1
            if arr_appear[phase] > max_c:
                max_c = arr_appear[phase]
                max_p = phase

    return (max_p, max_c)


def transform_factory(split, dataset):
    if split == 'train':
        # we don't add crop augumentation for the tradeoff of timecost.
        return Compose([
                        RandomCrop(size = 224),
                        ToTensor(),
                        Normalize(config[dataset]['mean'], config[dataset]['std']),
                        ])
    else:
        # tencrop for test.
        return Compose([
            FiveCrop(size=224),
            Lambda(lambda crops: [ToTensor()(crop) for crop in crops]),
            Lambda(lambda crops: torch.stack(
                [Normalize(config[dataset]['mean'], config[dataset]['std'])(crop) for crop in crops])),
        ])


def load_model(model, path):
    '''
    :param model: model class.
    :param path: the path for the pretrained model.
    :return: the model with the pretrained params.
    '''
    pretrained_dict = torch.load(path).state_dict()
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    # print(pretrained_dict.keys())
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


def annotation_to_dict(ground_truth_file):
    '''
    read the ground truth file.
    :param ground_truth_file: the ground truth file
    :return: the labels for each image in the video.
    '''
    ground_truth_dict = {}
    with open(ground_truth_file) as f:
        lines = f.readlines() # index Phase\n

    for line in lines:
        key, value = line.strip().split('\t')
        ground_truth_dict[key] = value
    return ground_truth_dict

def need_mapping(result_record, index, neg_index, interval, bound):
    for local_num in range(interval, bound, interval):
        mode = get_mode(result_record[max(0, index - local_num):index:])[0]
        if mode != neg_index:
            return False
    return True


def mapper_feature(l, global_num, neg):
    if len(l) == 0:
        l = [1]  # manually assign

    global_features = []
    global_phase_rate = []
    global_phase_dict = {i: 0 for i in range(neg)}
    global_phase_count = 1
    if len(l) < global_num:
        for i in range(global_num - len(l)):
            global_features.append((0, 0))
            global_phase_rate.append([0 for i in range(neg)])
        global_features += list(zip([0 if i == neg else i for i in l], [1] * len(l)))

        for c, phase in enumerate(l):
            if phase == neg:
                phase = 0
            global_phase_dict[phase] += 1
            global_phase_rate.append([global_phase_dict[k] / (c + 1) for k in range(neg)])

    else:
        interval = len(l) / global_num
        for i in range(global_num):
            left = int(i * interval)
            right = int((i + 1) * interval)
            mode, count = get_mode(l[left:right])
            # count phase --------------
            for phase in l[left:right]:
                if phase == neg:
                    phase = 0
                global_phase_dict[phase] += 1
                global_phase_count += (right - left)

            global_phase_rate.append([global_phase_dict[k] / (global_phase_count) for k in range(neg)])
            if mode == neg:
                global_features.append((0, count))
            else:
                global_features.append((mode, count))

    for i in range(len(global_features)):
        one_hot_temp = [0 for j in range(neg)]
        if global_features[i][0] != 0:
            one_hot_temp[global_features[i][0]] = 1
        one_hot_temp += global_phase_rate[i]
        global_features[i] = one_hot_temp

    if len(global_features) != global_num:
        print('error!')

    return np.array(global_features).astype('float32')


def mapper_data_generate(root, dataset, method, neg, global_num, interval, bound):
    global_feature_floder = os.path.join(root, 'global_feature')
    mapper_gt = os.path.join(root, 'mapper_gt')

    if not os.path.exists(global_feature_floder):
        os.makedirs(global_feature_floder)
    if not os.path.exists(mapper_gt):
        os.makedirs(mapper_gt)

    gt_path = os.path.join(root, 'annotation_folder')
    source_path = os.path.join(root, 'resnet')
    for file in os.listdir(gt_path):
        if not os.path.exists(os.path.join(global_feature_floder, file.split('.')[0])):
            os.makedirs(os.path.join(global_feature_floder, file.split('.')[0]))
        mapper_gts = []
        with open(os.path.join(source_path, file)) as f:
            if method == 'train':
                lines = [line.strip() for line in f.readlines()]  # the first line is frame\tphase\n
                lines = [config[dataset]['mapping_dict'][line.split('\t')[-1]] + 1 for line in lines]

            else:
                lines = [line.strip() for line in f.readlines()[1:]]  # the first line is frame\tphase\n
                lines = [int(line.split('\t')[1]) + 1 for line in lines]

        with open(os.path.join(gt_path, file)) as f:
            gt_lines = [line.strip() for line in f.readlines()]
            gt_lines = [config[dataset]['mapping_dict'][line.split('\t')[-1]] + 1 for line in
                        gt_lines]  # +1 since label start at 0, while 0 is the padding
        assert len(lines) == len(gt_lines)

        # before adding noise
        origin_lines = lines.copy()

        for i in range(len(origin_lines)):
            if origin_lines[i] == neg:
                if method == 'train' or need_mapping(origin_lines, i, neg, interval, bound):
                    global_feature = mapper_feature(lines[0:i], global_num, neg=neg)
                    np.save('{}/{}.npy'.format(os.path.join(global_feature_floder, file.split('.')[0]), i),
                            global_feature)
                    mapper_gts.append('{}\t{}\n'.format(i, gt_lines[i]))
            else:
                # add noise
                if method == 'train':
                    if random.random() < 0.15:
                        lines[i] = random.randint(1, neg)

        with open(os.path.join(mapper_gt, file), 'w') as f:
            f.writelines(mapper_gts)

        print('{} done!'.format(file))

def simple_convert(file_folder, target_folder, dataset):
    '''
    convert the test_dataset/resnet
    :return:
    '''
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    for file in os.listdir(file_folder):
        with open(os.path.join(file_folder, file), 'r') as f:
            lines = f.readlines()[1:]
            lines = expand(lines, dataset, 5)

        with open(os.path.join(target_folder, file.replace('.txt', '_pred.txt')), 'w') as f:
            f.writelines(lines)


if __name__ == '__main__':
    mapper_data_generate('m2cai16-workflow-5/train_dataset', 'm2cai16-workflow-5', 'train', 9, 240,5,65)
    mapper_data_generate('m2cai16-workflow-5/test_dataset', 'm2cai16-workflow-5', 'test', 9, 240,5,65)
    mapper_data_generate('cholec80-workflow-5/train_dataset', 'cholec80-workflow-5', 'train', 8, 240, 10, 100)
    mapper_data_generate('cholec80-workflow-5/test_dataset', 'cholec80-workflow-5', 'test', 8, 240, 10, 100)

    # simple_convert('cholec80-workflow-5/test_dataset/resnet', 'test_result','cholec80-workflow-5')

