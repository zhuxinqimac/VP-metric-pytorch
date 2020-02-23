#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: train_val.py
# --- Creation Date: 24-02-2020
# --- Last Modified: Mon 24 Feb 2020 02:58:53 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Train and validate function
"""
import numpy as np
import pdb
import time
import torch
from utils import AverageMeter, save_checkpoint, accuracy, show_inputs_target


def train(train_loader, model, criterion, optimizer, epoch, train_logger,
          args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    forward_time = AverageMeter()
    backward_time = AverageMeter()
    step_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    np.random.seed()
    end = time.time()
    for i, (inputs, target) in enumerate(train_loader):
        show_inputs_target(inputs, target, result_dir=args.result_dir)
        data_time.update(time.time() - end)
        target = target.cuda()
        input_var = torch.autograd.Variable(inputs).cuda()
        target_var = torch.autograd.Variable(target)

        print('train input_var.size:', input_var.size())
        pdb.set_trace()

        start_forwarding = time.time()

        # compute output
        output = model(input_var)

        print('output.size:', output.size())
        pdb.set_trace()

        end_forwarding = time.time()
        forward_time.update(end_forwarding - start_forwarding)

        loss = criterion(output, target_var)
        print('loss size:', loss.size())

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data, inputs.size(0))
        top1.update(prec1, inputs.size(0))
        top5.update(prec5, inputs.size(0))

        # compute gradient and do SGD step
        start_backwarding = time.time()
        optimizer.zero_grad()
        loss.backward()
        end_backwarding = time.time()
        backward_time.update(end_backwarding - start_backwarding)

        start_stepping = time.time()
        optimizer.step()
        end_stepping = time.time()
        step_time.update(end_stepping - start_stepping)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 30 == 0:
            log_line = (
                'Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Forward {forward_time.val:.7f} ({forward_time.avg:.7f})\t'
                'Backward {backward_time.val:.7f} ({backward_time.avg:.7f})\t'
                'Step {step_time.val:.7f} ({step_time.avg:.7f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch,
                    i,
                    len(train_loader),
                    batch_time=batch_time,
                    forward_time=forward_time,
                    backward_time=backward_time,
                    step_time=step_time,
                    data_time=data_time,
                    loss=losses,
                    top1=top1,
                    top5=top5,
                    lr=optimizer.param_groups[-1]['lr']))
            print(log_line)
            with open(train_logger, 'a') as f:
                f.write(log_line + '\n')

        if (i + 1) % 2000 == 0:
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_prec1': None,
                },
                is_best=False,
                result_dir=args.result_dir,
                filename='ep_' + str(epoch) + 'iter_' + str(i) +
                '_checkpoint.pth.tar')


def validate(val_loader, model, criterion, val_logger, epoch):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    forward_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (inputs, target, _) in enumerate(val_loader):
        if i == 200:
            batch_time.reset()
            data_time.reset()
            forward_time.reset()

        data_time.update(time.time() - end)
        start_copy_gpu = time.time()
        target = target.cuda()
        print('inputs.size:', inputs.size())
        pdb.set_trace()

        input_var = torch.autograd.Variable(inputs, volatile=True).cuda()
        target_var = torch.autograd.Variable(target, volatile=True)

        input_size = inputs.size()

        torch.cuda.synchronize()
        start_forwarding = time.time()
        with torch.no_grad():
            output = (model(input_var))
        torch.cuda.synchronize()
        end_forwarding = time.time()
        forward_time.update(end_forwarding - start_forwarding)

        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.data, inputs.size(0))
        top1.update(prec1, inputs.size(0))
        top5.update(prec5, inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 30 == 0:
            log_line = (
                'Test: Epoch:[{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Forward {forward_time.val:.7f} ({forward_time.avg:.7f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch,
                    i,
                    len(val_loader),
                    batch_time=batch_time,
                    forward_time=forward_time,
                    data_time=data_time,
                    loss=losses,
                    top1=top1,
                    top5=top5))
            print(log_line)
            with open(val_logger, 'a') as f:
                f.write(log_line + '\n')

    log_line = ('Testing Results: Prec@1 {top1.avg:.3f} '
                'Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'.format(
                    top1=top1, top5=top5, loss=losses))
    print(log_line)
    with open(val_logger, 'a') as f:
        f.write(log_line + '\n\n')

    return top1.avg
