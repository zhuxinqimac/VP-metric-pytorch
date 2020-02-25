#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: parser.py
# --- Creation Date: 24-02-2020
# --- Last Modified: Tue 25 Feb 2020 16:26:55 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Parser for VP metrics
"""

import argparse


def init_parser():
    parser = argparse.ArgumentParser(description='VP metrics.')
    parser.add_argument('--result_dir',
                        help='Results directory.',
                        type=str,
                        default='/mnt/hdd/repo_results/VP-metrics-pytorch')
    parser.add_argument('--data_dir',
                        help='Dataset directory.',
                        type=str,
                        default='/mnt/hdd/Datasets/test_data')
    parser.add_argument('--no_gpu',
                        help='Do not use GPUs.',
                        action='store_true')
    parser.add_argument('--in_channels',
                        help='Num channels for model input.',
                        type=int,
                        default=6)
    parser.add_argument('--out_dim',
                        help='Num output dimension.',
                        type=int,
                        default=7)
    parser.add_argument('--lr',
                        help='Learning rate.',
                        type=float,
                        default=0.01)
    parser.add_argument('--batch_size',
                        help='Batch size.',
                        type=int,
                        default=32)
    parser.add_argument('--epochs',
                        help='Num epochs to train.',
                        type=int,
                        default=60)
    parser.add_argument('--input_mode',
                        help='Input mode for model.',
                        type=str,
                        default='concat',
                        choices=['concat', 'diff'])
    parser.add_argument('--workers', help='Num workers.', type=int, default=4)
    return parser
