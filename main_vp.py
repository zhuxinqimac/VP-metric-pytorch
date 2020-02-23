#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: main_vp.py
# --- Creation Date: 24-02-2020
# --- Last Modified: Mon 24 Feb 2020 02:52:29 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
VP metrcs.
Train [x1, x2] --> [\delta z]
"""

import os
import pdb
import torch

import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from model import VarPred
from utils import worker_init_fn, split_indices, save_checkpoint
from parser_config import init_parser
from train_val import train, validate
from pair_dataset import PairDataset


def main():
    parser = init_parser()
    args = parser.parse_args()

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    model = VarPred(in_channels=args.in_channels, out_dim=args.out_dim)

    model.cuda()
    model = nn.DataParallel(model)
    cudnn.benchmark = True
    # optimizer = torch.optim.SGD(model.module.parameters(),
    # lr=args.lr,
    # momentum=0.9)
    optimizer = torch.optim.Adam(model.module.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss().cuda()

    train_list, test_list = split_indices(args.data_dir)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_loader = torch.utils.data.DataLoader(PairDataset(
        args.data_dir,
        train_list,
        image_tmpl='pair_%6d.jpg',
        transform=transform),
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               worker_init_fn=worker_init_fn)
    test_loader = torch.utils.data.DataLoader(PairDataset(
        args.data_dir,
        test_list,
        image_tmpl='pair_%6d.jpg',
        transform=transform),
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=args.workers,
                                              pin_memory=True,
                                              worker_init_fn=worker_init_fn)
    train_logger = os.path.join(args.result_dir, 'train.log')
    val_logger = os.path.join(args.result_dir, 'val.log')

    for epoch in range(args.epochs):
        # adjust_learning_rate(optimizer, epoch, args.lr_steps)

        # train for one epoch
        train(train_loader,
              model,
              criterion,
              optimizer,
              epoch,
              train_logger=train_logger,
              args=args)
        with open(train_logger, 'a') as f:
            f.write('\n')

        save_checkpoint(state={
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        },
                        is_best=False,
                        result_dir=args.result_dir,
                        filename='ep_' + str(epoch) + '_checkpoint.pth.tar')

        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            prec1 = validate(test_loader,
                             model,
                             criterion,
                             val_logger=val_logger,
                             epoch=epoch)

            # remember best prec@1 and save checkpoint
            if prec1 > best_prec1:
                is_best = prec1 > best_prec1
                best_prec1 = max(prec1, best_prec1)
                save_checkpoint(
                    {
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'best_prec1': best_prec1,
                    },
                    is_best=is_best,
                    result_dir=args.result_dir)


if __name__ == "__main__":
    main()
