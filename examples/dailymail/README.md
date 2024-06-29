# DailyMail & T5 Example

This directory contains scripts for fine-tuning RoBERTa computing influence scores on the SWAG dataset. The pipeline is motivated from [this HuggingFace Example](https://github.com/huggingface/transformers/tree/main/examples/pytorch/multiple-choice) demonstrates how to define `post_process_per_sample_gradient`.
To begin, install the necessary packages:

```bash
pip install -r requirements.txt
```

## Training

We will be using the pre-trained model.

## Computing Pairwise Influence Scores

To calculate pairwise influence scores on 10 query data points using `ekfac`, run:

```bash
python analyze.py --query_batch_size 100 \
    --train_batch_size 128 \
    --use_half_precision \
    --checkpoint_dir ./checkpoints \
    --factor_strategy ekfac
```

Alternative options for `factor_strategy` include `identity`, `diagonal`, and `kfac`. On an A100 (80GB), computing the pairwise scores (including EKFAC factors) takes approximately 4 hours:

```
----------------------------------------------------------------------------------------------------------------------------------
|  Action                       |  Mean duration (s)    |  Num calls            |  Total time (s)       |  Percentage %         |
----------------------------------------------------------------------------------------------------------------------------------
|  Total                        |  -                    |  11                   |  1.5434e+04           |  100 %                |
----------------------------------------------------------------------------------------------------------------------------------
|  Compute Pairwise Score       |  1.3667e+04           |  1                    |  1.3667e+04           |  88.546               |
|  Fit Lambda                   |  1241.0               |  1                    |  1241.0               |  8.0407               |
|  Fit Covariance               |  482.33               |  1                    |  482.33               |  3.125                |
|  Perform Eigendecomposition   |  22.946               |  1                    |  22.946               |  0.14867              |
|  Save Eigendecomposition      |  6.8322               |  1                    |  6.8322               |  0.044266             |
|  Save Covariance              |  6.6378               |  1                    |  6.6378               |  0.043006             |
|  Save Pairwise Score          |  2.6291               |  1                    |  2.6291               |  0.017034             |
|  Save Lambda                  |  2.3685               |  1                    |  2.3685               |  0.015346             |
|  Load Covariance              |  1.4838               |  1                    |  1.4838               |  0.0096133            |
|  Load Eigendecomposition      |  1.3845               |  1                    |  1.3845               |  0.0089699            |
|  Load All Factors             |  0.2517               |  1                    |  0.2517               |  0.0016308            |
----------------------------------------------------------------------------------------------------------------------------------
```

For more efficient computation, use Distributed Data Parallel (DDP):

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=2 analyze.py --factor_batch_size 128 \
    --query_batch_size 100 \
    --train_batch_size 128 \
    --use_half_precision \
    --checkpoint_dir ./checkpoints \
    --factor_strategy ekfac \
    --use_ddp
```

The overall computation time reduces to xx hours using 2 A100 GPUs:

```

```

## Evaluating Linear Datamodeling Score

The `evaluate_lds.py` script computes the [linear datamodeling score (LDS)](https://arxiv.org/abs/2303.14186). It measures the LDS obtained by 
retraining the network 500 times with different subsets of the dataset (5 repeats and 100 masks). We obtain `0.30` LDS.

```
Query Data Example:
 Option 0: <s>He looks disgusted and spits it out onto the plate.</s></s>He slides both hands around the crack.</s><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>
 Option 1: <s>He looks disgusted and spits it out onto the plate.</s></s>He passes someone to the bald guy.</s><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>
 Option 2: <s>He looks disgusted and spits it out onto the plate.</s></s>He picks up a piece of bread.</s><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>
 Option 3: <s>He looks disgusted and spits it out onto the plate.</s></s>He walks into the kitchen.</s><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>
 Label: 3

Top Influential Example:
 Option 0: <s>He lowers her hair back over the cut.</s></s>He lies fully clothed, still gazing at her scooter.</s><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>
 Option 1: <s>He lowers her hair back over the cut.</s></s>He bangs her head against her headrest.</s><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>
 Option 2: <s>He lowers her hair back over the cut.</s></s>He goes to the kitchen.</s><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>
 Option 3: <s>He lowers her hair back over the cut.</s></s>He gives him a sidelong look.</s><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>
 Label: 2
```