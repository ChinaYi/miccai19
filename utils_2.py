# -*- coding: utf-8 -*-
'''
Created on Sep 21, 2018

@author: yi
'''
from __future__ import print_function
from __future__ import division
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import *
from PIL import Image

from cfg import *



def _barcode(gt, predict, save_file = None):

    color_map = plt.cm.tab20
    axprops = dict(xticks=[], yticks=[], frameon=False)
    barprops = dict(aspect='auto', cmap = color_map,
                interpolation='nearest', vmin=0, vmax= 20)
    fig = plt.figure(figsize=(15, 6))
    # a horizontal barcode
    if gt is not None:
        ax1 = fig.add_axes([0, 0.6, 1, 0.15])
        ax1.set_title('Ground Truth')
        ax1.imshow([gt], **barprops)

    if predict is not None:
        ax2 = fig.add_axes([0, 0.2, 1, 0.15])
        ax2.set_title('Predicted')
        ax2.imshow([predict], **barprops)

        
    if save_file is not None:
        fig.savefig(save_file, dpi=400)
    else:
        plt.show()

    plt.close(fig)

def plot_barcode(eva_folder, dataset, savepath):
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    for file in os.listdir(eva_folder):
        if '_pred' in file:
            if dataset == 'm2cai16-workflow-5':
                gt_file = file.replace('_pred','')
            else:
                gt_file = file.replace('_pred', '-phase')
            gts = []
            preds = []

            with open(os.path.join(eva_folder, gt_file), 'r') as f:
                lines = f.readlines()[1:]
                lines = [line.strip().split('\t')[1] for line in lines]
                gts = [config[dataset]['mapping_dict'][line] for line in lines]

            with open(os.path.join(eva_folder, file), 'r') as f:
                lines = f.readlines()[1:]
                lines = [line.strip().split('\t')[1] for line in lines]
                preds = [config[dataset]['mapping_dict'][line] for line in lines]
            _barcode(gts, preds, os.path.join(savepath, file.replace('.txt','.jpg')))

if __name__ == '__main__':
    # plot_barcode('m2cai16-workflow-5/eva/test_dataset', 'm2cai16-workflow-5', 'vis/m2cai16')
    plot_barcode('cholec80-workflow-5/eva/test_dataset', 'cholec80-workflow-5', 'vis/cholec80')