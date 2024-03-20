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
On A100 (80GB), it takes approximately 12 hours to compute the pairwise scores (including computing EKFAC factors).

We can also use query batching to compute influence scores with larger query batch size.
```bash
python analyze.py --dataset_dir /mfs1/datasets/imagenet_pytorch/ \
    --query_gradient_rank 32 \
    --query_batch_size 500 \
    --train_batch_size 512 \
    --factor_strategy ekfac
```


## Computing Pairwise Influence Scores with DDP

You can also use DistributedDataParallel to speed up influence computations.
```bash
torchrun --standalone --nnodes=1 --nproc-per-node=4 ddp_analyze.py
```