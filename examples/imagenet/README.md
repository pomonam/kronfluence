```bash
torchrun --standalone --nnodes=1 --nproc-per-node=4 ddp_analyze.py
```


# ImageNet & ResNet-50 Example

This directory contains scripts for training ResNet-50 on ImageNet. 

## Training

We will use the pre-trained dataset from `torchvision.models.resnet50`.

## Computing Pairwise Influence Scores

To obtain a pairwise influence scores on 2000 query data points using `ekfac`, run the following command:
```bash
python analyze.py --dataset_dir /mfs1/datasets/imagenet_pytorch/ \
    --query_gradient_rank None \
    --query_batch_size 100 \
    --train_batch_size 128 \
    --factor_strategy ekfac
```
You can also use `identity`, `diagonal`, and `kfac`. On A100 (80GB), it takes roughly 1.5 minutes to compute the 
pairwise scores (including computing EKFAC factors).

## Mislabeled Data Detection

First, train the model with 10% of training dataset mislabeled by running the following command:
```bash
python train.py --dataset_dir ./data \
    --corrupt_percentage 0.1 \
    --checkpoint_dir ./checkpoints \
    --train_batch_size 512 \
    --eval_batch_size 1024 \
    --learning_rate 0.4 \
    --weight_decay 0.0001 \
    --num_train_epochs 25 \
    --seed 1004
```

Then, compute self-influence scores with the following command:
```bash
python detect_mislabeled_dataset.py --dataset_dir ./data \
    --corrupt_percentage 0.1 \
    --checkpoint_dir ./checkpoints \
    --factor_strategy ekfac
```

On A100 (80GB), it takes roughly 1.5 minutes to compute the self-influence scores (including computing EKFAC factors).
We can detect around 82% of mislabeled data points by inspecting 10% of the dataset using self-influence scores
(96% by inspecting 20%).