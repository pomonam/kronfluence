# CIFAR-10 & ResNet-9 Example

This directory contains scripts for training ResNet-9 and computing influence scores on CIFAR-10 dataset. The pipeline is motivated from 
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

To compute pairwise influence scores on 2000 query data points using the `ekfac` strategy, run the following command:

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

To use AMP when computing influence scores, run:

```bash
python analyze.py --query_batch_size 1000 \
    --dataset_dir ./data \
    --checkpoint_dir ./checkpoints \
    --factor_strategy ekfac \
    --use_half_precision
```

This reduces computation time to about 40 seconds on an A100 (80GB) GPU:

```
----------------------------------------------------------------------------------------------------------------------------------
|  Action                       |  Mean duration (s)    |  Num calls            |  Total time (s)       |  Percentage %         |
----------------------------------------------------------------------------------------------------------------------------------
|  Total                        |  -                    |  11                   |  42.316               |  100 %                |
----------------------------------------------------------------------------------------------------------------------------------
|  Compute Pairwise Score       |  19.565               |  1                    |  19.565               |  46.235               |
|  Fit Lambda                   |  9.173                |  1                    |  9.173                |  21.677               |
|  Fit Covariance               |  7.3723               |  1                    |  7.3723               |  17.422               |
|  Perform Eigendecomposition   |  2.6613               |  1                    |  2.6613               |  6.2891               |
|  Save Pairwise Score          |  2.0156               |  1                    |  2.0156               |  4.7633               |
|  Save Covariance              |  0.71699              |  1                    |  0.71699              |  1.6944               |
|  Save Eigendecomposition      |  0.52561              |  1                    |  0.52561              |  1.2421               |
|  Load Covariance              |  0.15732              |  1                    |  0.15732              |  0.37177              |
|  Save Lambda                  |  0.063394             |  1                    |  0.063394             |  0.14981              |
|  Load Eigendecomposition      |  0.051395             |  1                    |  0.051395             |  0.12146              |
|  Load All Factors             |  0.014144             |  1                    |  0.014144             |  0.033425             |
----------------------------------------------------------------------------------------------------------------------------------
```

You can run `half_precision_analysis.py` to verify that the scores computed with AMP have high correlations with those of the default configuration.

<p align="center">
<a href="#"><img width="380" img src="figure/half_precision.png" alt="Half Precision"/></a>
</p>

## Visualizing Influential Training Images

[This Colab notebook](https://colab.research.google.com/drive/1KIwIbeJh_om4tRwceuZ005fVKDsiXKgr?usp=sharing) provides a tutorial on visualizing the top influential training images.

## Mislabeled Data Detection

We can use self-influence scores (see **Section 5.4** for the [paper](https://arxiv.org/pdf/1703.04730.pdf)) to detect mislabeled examples. 
First, train the model with 10% of the training examples mislabeled by running:

```bash
python train.py --dataset_dir ./data \
    --corrupt_percentage 0.1 \
    --checkpoint_dir ./checkpoints \
    --train_batch_size 512 \
    --eval_batch_size 1024 \
    --learning_rate 0.4 \
    --weight_decay 0.001 \
    --num_train_epochs 25 \
    --seed 1004
```

Then, compute the self-influence scores with:

```bash
python detect_mislabeled_dataset.py --dataset_dir ./data \
    --corrupt_percentage 0.1 \
    --checkpoint_dir ./checkpoints \
    --factor_strategy ekfac
```

On an A100 (80GB) GPU, it takes roughly 2 minutes to compute the self-influence scores:

```
----------------------------------------------------------------------------------------------------------------------------------
|  Action                       |  Mean duration (s)    |  Num calls            |  Total time (s)       |  Percentage %         |
----------------------------------------------------------------------------------------------------------------------------------
|  Total                        |  -                    |  11                   |  122.28               |  100 %                |
----------------------------------------------------------------------------------------------------------------------------------
|  Compute Self-Influence Score |  61.999               |  1                    |  61.999               |  50.701               |
|  Fit Lambda                   |  34.629               |  1                    |  34.629               |  28.319               |
|  Fit Covariance               |  21.807               |  1                    |  21.807               |  17.833               |
|  Perform Eigendecomposition   |  1.8041               |  1                    |  1.8041               |  1.4754               |
|  Save Covariance              |  0.86378              |  1                    |  0.86378              |  0.70638              |
|  Save Eigendecomposition      |  0.84935              |  1                    |  0.84935              |  0.69458              |
|  Save Lambda                  |  0.18367              |  1                    |  0.18367              |  0.1502               |
|  Load Eigendecomposition      |  0.052867             |  1                    |  0.052867             |  0.043233             |
|  Load Covariance              |  0.051723             |  1                    |  0.051723             |  0.042298             |
|  Load All Factors             |  0.031986             |  1                    |  0.031986             |  0.026158             |
|  Save Self-Influence Score    |  0.010352             |  1                    |  0.010352             |  0.0084653            |
----------------------------------------------------------------------------------------------------------------------------------
```

Around 80% of mislabeled data points can be detected by inspecting 10% of the dataset (97% by inspecting 20%).

<p align="center">
<a href="#"><img width="380" img src="figure/mislabel.png" alt="Mislabeled Data Detection"/></a>
</p>