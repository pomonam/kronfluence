# ImageNet & ResNet-50 Example

This directory contains scripts for training ResNet-50 on ImageNet. 

## Training

We will use the pre-trained dataset from `torchvision.models.resnet50`.

## Computing Pairwise Influence Scores

To obtain a pairwise influence scores on 1000 query data points using `ekfac`, run the following command:
```bash
python analyze.py --dataset_dir /mfs1/datasets/imagenet_pytorch/ \
    --query_gradient_rank -1 \
    --query_batch_size 100 \
    --train_batch_size 256 \
    --factor_strategy ekfac
```
On A100 (80GB), it takes roughly 1.5 minutes to compute the pairwise scores (including computing EKFAC factors).


## Computing Pairwise Influence Scores with DDP

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=4 ddp_analyze.py
```