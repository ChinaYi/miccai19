import os
import numpy as np

import utils
from cfg import *


from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader



class VideoDataset_():
    def __init__(self, root, dataset, special_list, filter_type, feature_folder, ground_truth_folder):
        '''
        Return the images of one video with order perserved.
        :param root: root path for the dataset
        :param dataset: m2cai16-workflow-5 or cholec80-workflow-5
        :param special_list: list of video names, will be seperate from others by the filter_type
        :param filter_type: in or not_in
        :param feature_folder: image folder
        :param ground_truth_folder: ground_truth_folder
        '''
        self.video_labels = []
        self.video_img_names = []
        self.video_name = []

        img_folder = os.path.join(root, feature_folder)
        ground_truth_folder = os.path.join(root, ground_truth_folder)

        for video in os.listdir(img_folder):
            if video not in special_list and filter_type == 'in':
                continue
            if video in special_list and filter_type == 'not_in':
                continue

            self.video_name.append(video)

            annotation_dict = utils.annotation_to_dict(os.path.join(ground_truth_folder, '{}.txt'.format(video)))
            sorted_pics = []

            for img in os.listdir(os.path.join(img_folder, video)):
                img_index = img.split('.')[0]

                if img_index in annotation_dict.keys():
                    sorted_pics.append(
                        (os.path.join(img_folder, video, img),
                         config[dataset]['mapping_dict'][annotation_dict[img_index]], int(img_index))
                    )


            sorted_pics = sorted(sorted_pics, key=lambda a: a[2])
            self.video_img_names.append([a[0] for a in sorted_pics])
            self.video_labels.append([a[1] for a in sorted_pics])
            print('video {} has {} pics'.format(video, len(sorted_pics)))
        print('[INFO] length of dataset {} is {}'.format(root, len(self.video_labels)))

    def __len__(self):
        return len(self.video_labels)

    def __getitem__(self, start, end):
        return self.video_img_names[start:end], self.video_labels[start:end], self.video_name[start:end]


class VideoDataset(Dataset):
    def __init__(self, video_imgs, video_labels, transform):
        '''
        Construct use the _VideoDataset.__getitem__()
        :param video_imgs: img names that come from the same video
        :param video_labels: img labels that come from the same video
        :param transform: transform for image
        '''
        self.imgs, self.labels = self._merge(video_imgs, video_labels)
        self.transform = transform
        print('[INFO] length of dataset is {}'.format(len(self.imgs)))

    def __len__(self):
        return len(self.imgs)

    def _merge(self, video_imgs, video_labels):
        imgs = []
        labels = []

        for video_index in range(len(video_imgs)):
            imgs += video_imgs[video_index]
            labels += video_labels[video_index]
        return imgs, labels

    def __getitem__(self, index):
        img, img_name, label = default_loader(self.imgs[index]), self.imgs[index], self.labels[index]
        if self.transform is not None:
            img = self.transform(img)

        return img, label, img_name


class MapperDataset(Dataset):
    def __init__(self, dataset_path, filter, filter_type, prefix):
        bin_feature_floder = os.path.join(dataset_path, prefix, 'global_feature')
        gt_floder = os.path.join(dataset_path, prefix, 'mapper_gt')
        img_feature_floder = os.path.join(dataset_path, prefix, 'feature_folder')

        self.bin_features = []
        self.gts = []
        self.img_features = []

        for video in os.listdir(bin_feature_floder):
            if video in filter and filter_type == 'not_in':
                continue
            if video not in filter and filter_type == 'in':
                continue
            gt_path = os.path.join(gt_floder, video + '.txt')

            with open(gt_path) as f:
                gt_lines = f.readlines()
            for i in range(len(gt_lines)):
                prefix, label = gt_lines[i].strip().split('\t')
                self.gts.append(int(label))

                bin_full_name = os.path.join(bin_feature_floder, video, prefix + '.npy')
                bin_feature = np.load(bin_full_name)
                self.bin_features.append(bin_feature)

                img_feature_full_name = os.path.join(img_feature_floder, video, prefix + '.npy')
                self.img_features.append(np.load(img_feature_full_name))

            print('Loading video {} Done!'.format(video))

    def __len__(self):
        return len(self.bin_features)

    def __getitem__(self, index):
        return (self.bin_features[index],
                self.img_features[index],
                self.gts[index])
