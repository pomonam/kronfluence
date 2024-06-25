# CIFAR-10 & ResNet-9 Example

This directory contains scripts for training ResNet-9 on CIFAR-10. The pipeline is motivated from 
[TRAK repository](https://github.com/MadryLab/trak/blob/main/examples/cifar_quickstart.ipynb). To get started, please install the necessary packages by running the following command:

```bash
pip install -r requirements.txt
```

## Training

To train ResNet-9 on the CIFAR-10 dataset, run the following command:

```bash
python train.py --dataset_dir ./data \
    --checkpoint_dir ./checkpoints \
    --train_batch_size 512 \
    --eval_batch_size 1024 \
    --learning_rate 0.4 \
    --weight_decay 0.001 \
    --num_train_epochs 25 \
    --seed 1004
```

This will train the model using the specified hyperparameters and save the trained checkpoint in the `./checkpoints` directory.

## Computing Pairwise Influence Scores

To compute pairwise influence scores on 2000 query data points using the `ekfac` factorization strategy, run the following command:

```bash
python analyze.py --query_batch_size 1000 \
    --dataset_dir ./data \
    --checkpoint_dir ./checkpoints \
    --factor_strategy ekfac
```

In addition to `ekfac`, you can also use `identity`, `diagonal`, and `kfac` as the `factor_strategy`. On an A100 (80GB) GPU, it takes roughly 2 minutes to compute the pairwise scores (including computing the EKFAC factors):

```
----------------------------------------------------------------------------------------------------------------------------------
|  Action                       |  Mean duration (s)    |  Num calls            |  Total time (s)       |  Percentage %         |
----------------------------------------------------------------------------------------------------------------------------------
|  Total                        |  -                    |  11                   |  112.83               |  100 %                |
----------------------------------------------------------------------------------------------------------------------------------
|  Compute Pairwise Score       |  47.989               |  1                    |  47.989               |  42.532               |
|  Fit Lambda                   |  34.639               |  1                    |  34.639               |  30.7                 |
|  Fit Covariance               |  21.841               |  1                    |  21.841               |  19.357               |
|  Save Pairwise Score          |  3.5998               |  1                    |  3.5998               |  3.1905               |
|  Perform Eigendecomposition   |  2.7724               |  1                    |  2.7724               |  2.4572               |
|  Save Covariance              |  0.85695              |  1                    |  0.85695              |  0.75951              |
|  Save Eigendecomposition      |  0.85628              |  1                    |  0.85628              |  0.75892              |
|  Save Lambda                  |  0.12327              |  1                    |  0.12327              |  0.10925              |
|  Load Eigendecomposition      |  0.056494             |  1                    |  0.056494             |  0.05007              |
|  Load All Factors             |  0.048981             |  1                    |  0.048981             |  0.043412             |
|  Load Covariance              |  0.046798             |  1                    |  0.046798             |  0.041476             |
----------------------------------------------------------------------------------------------------------------------------------
```

To use AMP when computing influence scores (in addition to half precision when computing influence factors and scores), run:

```bash
python analyze.py --query_batch_size 1000 \
    --dataset_dir ./data \
    --checkpoint_dir ./checkpoints \
    --factor_strategy ekfac \
    --use_half_precision
```

## Mislabeled Data Detection

We can use self-influence scores (see Section 5.4 for the [paper](https://arxiv.org/pdf/1703.04730.pdf)) to detect mislabeled examples. 
First, train the model with 10% of training examples mislabeled by running the following command:
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

On A100 (80GB), it takes roughly 1.5 minutes to compute the self-influence scores.
We can detect around 82% of mislabeled data points by inspecting 10% of the dataset (96% by inspecting 20%).