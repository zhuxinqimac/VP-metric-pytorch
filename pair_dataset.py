#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: pair_dataset.py
# --- Creation Date: 24-02-2020
# --- Last Modified: Mon 24 Feb 2020 03:55:07 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Image pair dataset
"""

import torch.utils.data as data
import torch

from PIL import Image
import os
import os.path
import numpy as np
import pdb

from multiprocessing.dummy import Pool as ThreadPool


class PairDataset(data.Dataset):
    # PairDataset(
    # args.data_dir, train_list, image_tmpl='pair_%6d.jpg', transform=transform)
    def __init__(self,
                 data_dir,
                 idx_list,
                 image_tmpl='pair_{:06d}.jpg',
                 transform=None):

        self.data_dir = data_dir
        self.idx_list = idx_list
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.label_file = os.path.join(self.data_dir, 'labels.npy')
        self.labels = np.load(self.label_file)

    def __getitem__(self, hyper_idx):
        idx = self.idx_list[hyper_idx]
        img_name = os.path.join(self.data_dir, self.image_tmpl.format(idx))
        img_np = load_image(img_name)
        label = self.labels[idx]
        img_tensor = self.transform(img_np)
        return img_tensor, np.argmax(label)

    def __len__(self):
        return len(self.idx_list)


def load_image(filename):
    img = Image.open(filename).convert('RGB')
    img_np = np.array(img)  # np.array(h, w, c), (0, 255) (RGB)
    return img_np
