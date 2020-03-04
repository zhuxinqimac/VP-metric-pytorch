#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: generate_fid_dataset.py
# --- Creation Date: 04-03-2020
# --- Last Modified: Wed 04 Mar 2020 20:34:17 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Generate a dataset for FID score calculation
"""

import argparse
import os
import pdb
import glob
import numpy as np
from PIL import Image

def crop_images(args):
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    img_exts = ('*.jpg', '*.png')
    source_imgs_path = []
    for img_ext in img_exts:
        source_imgs_path.extend(
            sorted(glob.glob(
                os.path.join(args.source_dir, img_ext))))
    idxes = np.arange(len(source_imgs_path))
    np.random.shuffle(idxes)
    for i in range(args.n_imgs):
        source_path = source_imgs_path[idxes[i]]
        img = Image.open(source_path)
        img_x, img_y = img.size
        x_s = (img_x - args.crop_w) // 2
        y_s = (img_y - args.crop_h) // 2
        cropped_img = img.crop((x_s, y_s, x_s + args.crop_w, y_s + args.crop_h))
        img_name = os.path.basename(source_path)
        save_path = os.path.join(args.result_dir, img_name)
        cropped_img.save(save_path)

def main():
    parser = argparse.ArgumentParser(description='Project description.')
    parser.add_argument('--result_dir', help='Results directory.',
                        type=str, default='/mnt/hdd/repo_results/test')
    parser.add_argument('--data_dir', help='Dataset directory.',
                        type=str, default='/mnt/hdd/Datasets/test_data')
    parser.add_argument('--img_type', help='Image type directory.',
                        type=str, default='/mnt/hdd/Datasets/test_data')
    parser.add_argument('--crop_h', help='Cropped height.',
                        type=int, default=128)
    parser.add_argument('--crop_w', help='Cropped width.',
                        type=int, default=128)
    parser.add_argument('--n_imgs', help='Number of images to crop.',
                        type=int, default=50000)
    # parser.add_argument('--no_gpu', help='Do not use GPUs.',
                        # action='store_true')
    args = parser.parse_args()

    crop_images(args)


if __name__ == "__main__":
    main()
