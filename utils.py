#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: utils.py
# --- Creation Date: 24-02-2020
# --- Last Modified: Mon 24 Feb 2020 04:25:01 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Utils for VP metrics
"""

import os
import torch
import numpy as np
import shutil
import torchvision
from PIL import Image


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def split_indices(data_dir, test_ratio):
    label_file = os.path.join(data_dir, 'labels.npy')
    labels = np.load(label_file)
    n_data = labels.shape[0]
    shuffled = np.arange(n_data)
    np.random.shuffle(shuffled)
    test_list = shuffled[:n_data * test_ratio]
    train_list = shuffled[n_data * test_ratio:]
    return train_list, test_list


def save_checkpoint(state, is_best, result_dir, filename='tmp.pth.tar'):
    if is_best:
        print('Saving best checkpoint...')
        filename = 'model_best.pth.tar'
        torch.save(state, os.path.join(result_dir, filename))
        with open(os.path.join(result_dir, 'best_epoch.txt'), 'a') as f:
            f.write('best epoch: ' + str(state['epoch']))
    else:
        print('Saving checkpoint...')
        torch.save(state, os.path.join(result_dir, filename))


def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def show_inputs_target(inputs, target, result_dir):
    img = torchvision.utils.make_grid(inputs)
    img = img / 2 + 0.5  # unnormalize
    img_np = img.numpy()
    img_np = (np.transpose(img_np, (1, 2, 0)) * 255).astype(np.uint8)
    img = Image.fromarray(img_np)
    img.save(os.path.join(result_dir, 'sainity.jpg'))
    print('labels:', str(target))
