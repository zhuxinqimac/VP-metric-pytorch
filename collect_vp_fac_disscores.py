import os
import numpy as np
import argparse
import glob
import matplotlib.pyplot as plt


def plot_relation(fac_accs, vp_accs, config_names, args):
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    for i in range(len(fac_accs)):
        plt.plot(fac_accs[i], vp_accs[i], 'ro')
        # if fac_accs[i] < 0.8:
        plt.text(fac_accs[i], vp_accs[i], config_names[i], fontsize=8)
    filename = 'fac_vp_factype_' + args.fac_dis_type + \
            '_vptype_' + args.vp_dis_type + \
            '_vpepoch_' + str(args.target_epoch) + '.jpg'
    plt.savefig(os.path.join(args.result_dir, filename))


def save_points(fac_accs, vp_accs, config_names, args):
    filename = args.save_points_name + '.txt'
    with open(os.path.join(args.result_dir, filename), 'w') as f:
        f.write(args.save_points_name + '\n')
        for i in range(len(fac_accs)):
            f.write(
                str(fac_accs[i]) + ' : ' + str(vp_accs[i]) + ' : ' +
                config_names[i] + '\n')


def readlines_of(filename):
    with open(filename, 'r') as f:
        data = f.readlines()
    return data


def get_val_line(data, target_epoch):
    j = 0
    for i, line in enumerate(data):
        if line == '\n':
            if j == target_epoch:
                output = data[i - 1]
                break
            j += 1
    return output


def get_all_val_line(data, target_epoch):
    j = 0
    val_lines = []
    for i, line in enumerate(data):
        if line == '\n':
            val_lines.append(data[i - 1])
            if j >= target_epoch:
                break
            j += 1
    return val_lines


def get_dis_scores(data_dir, args):
    fac_acc_txt = os.path.join(data_dir, 'acc.txt')
    config_name = os.path.basename(data_dir)
    if not os.path.isfile(fac_acc_txt):
        return None, None, None
    data = readlines_of(fac_acc_txt)
    line = data[0]  # only one line in this file
    if args.fac_dis_type == 'last':
        if args.pretrained_fac:
            fac_acc = float(line.strip().split(':')[-1])
        else:
            fac_acc = float(line.strip().split()[-3].split(';')[0])
    elif args.fac_dis_type == 'best':
        if args.pretrained_fac:
            fac_acc = float(line.strip().split(':')[-1])
        else:
            fac_acc = float(line.strip().split()[-1].split(':')[1])
    else:
        raise ValueError('Not supported fac_dis_type: ' + args.fac_dis_type)

    vp_acc_txt = os.path.join(data_dir, 'pairs_train', 'val.log')
    if not os.path.isfile(vp_acc_txt):
        return None, None, None
    if args.vp_dis_type == 'best':
        vp_acc_bestepoch_txt = os.path.join(data_dir, 'pairs_train',
                                            'best_epoch.txt')
        data = readlines_of(vp_acc_bestepoch_txt)
        line = data[0]
        target_epoch = int(line.strip().split()[-1]) - 1
        data = readlines_of(vp_acc_txt)
        val_line = get_val_line(data, target_epoch)
        vp_acc = float(val_line.strip().split()[3])
    elif args.vp_dis_type == 'other':
        data = readlines_of(vp_acc_txt)
        val_line = get_val_line(data, args.target_epoch)
        vp_acc = float(val_line.strip().split()[3])
    elif args.vp_dis_type == 'avg':
        data = readlines_of(vp_acc_txt)
        val_lines = get_all_val_line(data, args.target_epoch)
        vp_accs = [float(val_line.strip().split()[3]) for val_line in val_lines]
        vp_acc = sum(vp_accs) / len(vp_accs)
    return fac_acc, vp_acc, config_name


def main():
    parser = argparse.ArgumentParser(description='Collect metrics data.')
    parser.add_argument('--result_dir',
                        help='Results directory.',
                        type=str,
                        default='/mnt/hdd/repo_results/VP-metrics-pytorch')
    parser.add_argument('--target_dir',
                        help='Target directory.',
                        type=str,
                        default='/mnt/hdd/Datasets/test_data')
    parser.add_argument('--fac_dis_type',
                        help='Factor disentangle metrics txt type.',
                        type=str,
                        default='last',
                        choices=['last', 'best'])
    parser.add_argument('--vp_dis_type',
                        help='VP disentangle metrics txt type.',
                        type=str,
                        default='best',
                        choices=['best', 'other', 'avg'])
    parser.add_argument('--target_epoch',
                        help='If vp_dis_type != best, ' +
                        'which epoch to use. Starting with 0.',
                        type=int,
                        default=60)
    parser.add_argument('--pretrained_fac',
                        help='If use pretrained fac disen score.',
                        action='store_true')
    parser.add_argument('--save_points_name',
                        help='The filename to save collected score points.',
                        type=str)

    args = parser.parse_args()

    data_dirs = glob.glob(os.path.join(args.target_dir, '64_*'))
    fac_accs = []
    vp_accs = []
    config_names = []
    for data_dir in data_dirs:
        fac_acc, vp_acc, config_name = get_dis_scores(data_dir, args)
        if fac_acc is None:
            continue
        fac_accs.append(fac_acc)
        vp_accs.append(vp_acc)
        config_names.append(config_name)

    plot_relation(fac_accs, vp_accs, config_names, args)
    save_points(fac_accs, vp_accs, config_names, args)


if __name__ == "__main__":
    main()
