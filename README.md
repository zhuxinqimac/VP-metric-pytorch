# Variation Predictability Metric

This repository contains the independent code for VP-metric in [Learning Disentangled Representations with Latent Variation Predictability].

## Requirements

* Numpy.
* PyTorch >= 1.3.1

## Training
Once you have a dataset of [(x1, x2) --> \delta z], you can use this code
to train a simple ConvNet to do the evaluation.

```
CUDA_VISIBLE_DEVICES=0 \
    python main_vp.py \
    --result_dir /path/to/result-dir \
    --data_dir /path/to/image-pair/dir \
    --in_channels 3 \
    --out_dim 30 \
    --lr 0.01 \
    --batch_size 32 \
    --epochs 200 \
    --input_mode diff \
    --test_ratio 0.9
```

Then use:
```
python get_best_score.py --target_dir /path/to/result-dir
```
to obtain the VP score.
