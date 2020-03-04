#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: get_best_score.py
# --- Creation Date: 29-02-2020
# --- Last Modified: Sat 29 Feb 2020 14:20:36 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Get the highest score from training results
"""
import os
import argparse
from collect_vp_fac_disscores import readlines_of, get_val_line


def get_dis_score(args):
    vp_acc_txt = os.path.join(args.target_dir, 'val.log')
    if not os.path.isfile(vp_acc_txt):
        return 0
    vp_acc_bestepoch_txt = os.path.join(args.target_dir, 'best_epoch.txt')
    data = readlines_of(vp_acc_bestepoch_txt)
    line = data[0]
    target_epoch = int(line.strip().split()[-1]) - 1
    data = readlines_of(vp_acc_txt)
    val_line = get_val_line(data, target_epoch)
    vp_acc = float(val_line.strip().split()[3])
    return vp_acc


def main():
    parser = argparse.ArgumentParser(description='Collect best score.')
    parser.add_argument('--target_dir',
                        help='Target directory.',
                        type=str,
                        default='/mnt/hdd/Datasets/test_data/pair_train')
    parser.add_argument('--vp_dis_type',
                        help='VP disentangle metrics txt type.',
                        type=str,
                        default='best')

    args = parser.parse_args()

    vp_acc = get_dis_score(args)
    print(vp_acc)


if __name__ == "__main__":
    main()
