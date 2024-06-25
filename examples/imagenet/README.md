# ImageNet & ResNet-50 Example

This directory contains scripts for computing influence scores on the ImageNet dataset using ResNet-50.
To get started, please install the necessary packages by running:

```bash
pip install -r requirements.txt
```

## Training

We will use the pre-trained ResNet-50 model from `torchvision.models.resnet50`.

## Computing Pairwise Influence Scores

To compute pairwise influence scores on 1000 query data points using the `ekfac` factorization strategy, run the following command:

```bash
python analyze.py --dataset_dir PATH_TO_IMAGENET \
    --query_gradient_rank -1 \
    --query_batch_size 100 \
    --train_batch_size 512 \
    --factor_strategy ekfac
```

Replace `PATH_TO_IMAGENET` with the path to your ImageNet dataset directory.
On an A100 (80GB) GPU, it takes approximately 11 hours to compute the pairwise scores (including computing EKFAC factors):

```

```

Query batching (low-rank approximation to the query gradient; see **Section 3.2.2** from the paper) can be used to compute influence scores with a larger query batch size:

```bash
python analyze.py --dataset_dir PATH_TO_IMAGENET \
    --query_gradient_rank 32 \
    --query_batch_size 500 \
    --train_batch_size 512 \
    --factor_strategy ekfac
```

On an A100 (80GB) GPU, it takes roughly 3.5 hours to compute the pairwise scores with query batching (including computing EKFAC factors):

```

```

Assuming you ran the above two commands, `query_batching_analysis.py` contains code to compute the correlations between the full-rank and low-rank scores.

<p align="center">
<a href="#"><img width="380" img src="figure/query_batching.png" alt="Counterfactual"/></a>
</p>

The averaged correlations between the low-rank and full rank scores for 100 data points is 0.95.
For more efficient computation, use half precision:

```bash
python analyze.py --dataset_dir PATH_TO_IMAGENET \
    --query_gradient_rank 32 \
    --query_batch_size 500 \
    --train_batch_size 512 \
    --factor_strategy ekfac \
    --use_half_precision
```

This reduces computation time to about 20 minutes on an A100 (80GB) GPU:

```
----------------------------------------------------------------------------------------------------------------------------------
|  Action                       |  Mean duration (s)    |  Num calls            |  Total time (s)       |  Percentage %         |
----------------------------------------------------------------------------------------------------------------------------------
|  Total                        |  -                    |  11                   |  1211.8               |  100 %                |
----------------------------------------------------------------------------------------------------------------------------------
|  Compute Pairwise Score       |  1034.5               |  1                    |  1034.5               |  85.368               |
|  Fit Lambda                   |  88.231               |  1                    |  88.231               |  7.2811               |
|  Fit Covariance               |  59.746               |  1                    |  59.746               |  4.9305               |
|  Perform Eigendecomposition   |  14.831               |  1                    |  14.831               |  1.2239               |
|  Save Covariance              |  5.8912               |  1                    |  5.8912               |  0.48617              |
|  Save Eigendecomposition      |  5.7726               |  1                    |  5.7726               |  0.47638              |
|  Save Lambda                  |  1.624                |  1                    |  1.624                |  0.13402              |
|  Load Covariance              |  0.34494              |  1                    |  0.34494              |  0.028465             |
|  Load Eigendecomposition      |  0.33595              |  1                    |  0.33595              |  0.027724             |
|  Load All Factors             |  0.26719              |  1                    |  0.26719              |  0.022049             |
|  Save Pairwise Score          |  0.26006              |  1                    |  0.26006              |  0.021461             |
----------------------------------------------------------------------------------------------------------------------------------
```


## Computing Pairwise Influence Scores with DDP

You can also use [DistributedDataParallel (DDP)](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) to speed up influence computations. To run influence analysis with four A100 (80GB) GPUs and query batching, use the command:

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=2 ddp_analyze.py --dataset_dir PATH_TO_IMAGENET \
    --query_gradient_rank 32 \
    --factor_batch_size 512 \
    --query_batch_size 100 \
    --train_batch_size 512 \
    --factor_strategy ekfac \
    --half_precision
```

It takes approximately 1 hour to compute the pairwise scores:

```
----------------------------------------------------------------------------------------------------------------------------------
|  Action                       |  Mean duration (s)    |  Num calls            |  Total time (s)       |  Percentage %         |
----------------------------------------------------------------------------------------------------------------------------------
|  Total                        |  -                    |  11                   |  3423.3               |  100 %                |
----------------------------------------------------------------------------------------------------------------------------------
|  Compute Pairwise Score       |  2917.2               |  1                    |  2917.2               |  85.215               |
|  Fit Lambda                   |  298.62               |  1                    |  298.62               |  8.7232               |
|  Fit Covariance               |  137.9                |  1                    |  137.9                |  4.0282               |
|  Save Pairwise Score          |  48.122               |  1                    |  48.122               |  1.4057               |
|  Perform Eigendecomposition   |  7.7503               |  1                    |  7.7503               |  0.2264               |
|  Save Eigendecomposition      |  5.9978               |  1                    |  5.9978               |  0.1752               |
|  Save Covariance              |  5.7442               |  1                    |  5.7442               |  0.1678               |
|  Save Lambda                  |  0.95602              |  1                    |  0.95602              |  0.027927             |
|  Load Covariance              |  0.5718               |  1                    |  0.5718               |  0.016703             |
|  Load Eigendecomposition      |  0.34755              |  1                    |  0.34755              |  0.010153             |
|  Load All Factors             |  0.13107              |  1                    |  0.13107              |  0.0038288            |
----------------------------------------------------------------------------------------------------------------------------------
```

You can use more GPUs to further speed up the influence computations.
