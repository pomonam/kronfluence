# ImageNet & ResNet-50 Example

This directory contains scripts for training ResNet-50 on ImageNet. 

## Training

We will use the pre-trained model from `torchvision.models.resnet50`.

## Computing Pairwise Influence Scores

To obtain a pairwise influence scores on 1000 query data points using `ekfac`, run the following command:
```bash
python analyze.py --dataset_dir /mfs1/datasets/imagenet_pytorch/ \
    --query_gradient_rank -1 \
    --query_batch_size 100 \
    --train_batch_size 512 \
    --factor_strategy ekfac
```
On A100 (80GB), it takes approximately 12 hours to compute the pairwise scores (including computing EKFAC factors).

We can also use query batching (low-rank approximation to the query gradient) to compute influence scores with a 
larger query batch size.
```bash
python analyze.py --dataset_dir /mfs1/datasets/imagenet_pytorch/ \
    --query_gradient_rank 32 \
    --query_batch_size 500 \
    --train_batch_size 512 \
    --factor_strategy ekfac
```
On A100 (80GB), it takes less than 4 hours to compute the pairwise scores with query batching (including computing EKFAC factors).

But how accurate are the low-rank approximations? Assuming that you ran above two commands, `query_batching_analysis.py`
contains code the compute the correlations between the full rank prediction and low-rank prediction.

<p align="center">
<a href="#"><img width="380" img src="figure/query_batching.png" alt="Counterfactual"/></a>
</p>
The averaged correlations between the low-rank and full rank for 100 data points is 0.95.


## Computing Pairwise Influence Scores with DDP

You can also use DistributedDataParallel (DDP) to speed up influence computations. You can run:
```bash
torchrun --standalone --nnodes=1 --nproc-per-node=2 ddp_analyze.py
```
On 2 A100 (80GB), it takes approximately 6 hours to compute the pairwise scores. When available, you can use more GPUs 
to speed up influence computations.
