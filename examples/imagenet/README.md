# ImageNet & ResNet-50 Example

This directory contains scripts for training ResNet-50 on ImageNet. Please begin by installing necessary packages.
```bash
pip install -r requirements.txt
```

## Training

We will use the pre-trained model from `torchvision.models.resnet50`.

## Computing Pairwise Influence Scores

To obtain a pairwise influence scores on 1000 query data points using `ekfac`, run the following command:
```bash
python analyze.py --dataset_dir PATH_TO_IMAGENET \
    --query_gradient_rank -1 \
    --query_batch_size 100 \
    --train_batch_size 512 \
    --factor_strategy ekfac
```
On A100 (80GB), it takes approximately 12 hours to compute the pairwise scores (including computing EKFAC factors).

We can also use query batching (low-rank approximation to the query gradient; see Section 3.2.2 from the [paper](https://arxiv.org/pdf/2308.03296.pdf)) to compute influence scores with a 
larger query batch size.
```bash
python analyze.py --dataset_dir PATH_TO_IMAGENET \
    --query_gradient_rank 32 \
    --query_batch_size 500 \
    --train_batch_size 512 \
    --factor_strategy ekfac
```
On A100 (80GB), it takes roughly 4 hours to compute the pairwise scores with query batching (including computing EKFAC factors).
Assuming that you ran above two commands, `query_batching_analysis.py`
contains code to compute the correlations between the full rank and low-rank scores.

<p align="center">
<a href="#"><img width="380" img src="figure/query_batching.png" alt="Counterfactual"/></a>
</p>
The averaged correlations between the low-rank and full rank scores for 100 data points is 0.95.

## Computing Pairwise Influence Scores with DDP

You can also use DistributedDataParallel (DDP) to speed up influence computations. You can run:
```bash
torchrun --standalone --nnodes=1 --nproc-per-node=2 ddp_analyze.py --dataset_dir PATH_TO_IMAGENET \
    --query_gradient_rank -1 \
    --factor_batch_size 512 \
    --query_batch_size 100 \
    --train_batch_size 512 \
    --factor_strategy ekfac
```
On 2 A100 (80GB), it takes approximately 6 hours to compute the pairwise scores. When available, you can use more GPUs 
to speed up influence computations.
