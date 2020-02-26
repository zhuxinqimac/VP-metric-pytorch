import os
import glob
import argparse
import shutil

def main():
    parser = argparse.ArgumentParser(description='Backup pairs_train folders.')
    parser.add_argument('--asset_dir',
                        help='Asset directory.',
                        type=str,
                        default='/mnt/hdd/repo_results/VP-metrics-pytorch')
    parser.add_argument('--dir_to_move',
                        help='The dir to move.',
                        type=str,
                        default='pairs_train')
    parser.add_argument('--new_idx',
                        help='Move pairs_train to a new index.',
                        type=int,
                        default=1)
    args = parser.parse_args()
    target_dirs = glob.glob(os.path.join(args.asset_dir, '*'))
    for i_dir in target_dirs:
        old_pairs_dir = os.path.join(i_dir, args.dir_to_move)
        if os.path.exists(old_pairs_dir):
            new_pairs_dir = os.path.join(i_dir, 
                    args.dir_to_move+'_'+str(args.new_idx))
            shutil.move(old_pairs_dir, new_pairs_dir)

if __name__ == "__main__":
    main()
