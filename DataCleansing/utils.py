from cfg import *

import torch
from torchvision.transforms import *

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
