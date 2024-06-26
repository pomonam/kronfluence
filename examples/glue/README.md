# GLUE & BERT Example

This directory contains scripts for fine-tuning BERT and computing influence scores on the GLUE benchmark. The pipeline is motivated from [this HuggingFace Example](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification).
To get started, please install the necessary packages:

```bash
pip install -r requirements.txt
```

## Training

To fine-tune BERT on a specific dataset, run the following command (we are using the `SST2` dataset in this example):

```bash
python train.py --dataset_name sst2 \
    --checkpoint_dir ./checkpoints \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --learning_rate 3e-05 \
    --weight_decay 0.01 \
    --num_train_epochs 3 \
    --seed 1004
```

This will save the trained checkpoint in the `./checkpoints` directory.

## Computing Pairwise Influence Scores

To obtain pairwise influence scores on a maximum of 2000 query data points using `ekfac`, run the following command:

```bash
python analyze.py --dataset_name sst2 \
    --query_batch_size 175 \
    --train_batch_size 128 \
    --checkpoint_dir ./checkpoints \
    --factor_strategy ekfac
```

On an A100 (80GB), it takes roughly 95 minutes to compute the pairwise scores for `SST2` with around 900 query data points (including computing EKFAC factors):

```
----------------------------------------------------------------------------------------------------------------------------------
|  Action                       |  Mean duration (s)    |  Num calls            |  Total time (s)       |  Percentage %         |
----------------------------------------------------------------------------------------------------------------------------------
|  Total                        |  -                    |  11                   |  5568.5               |  100 %                |
----------------------------------------------------------------------------------------------------------------------------------
|  Compute Pairwise Score       |  2668.0               |  1                    |  2668.0               |  47.913               |
|  Fit Lambda                   |  2361.5               |  1                    |  2361.5               |  42.408               |
|  Fit Covariance               |  483.63               |  1                    |  483.63               |  8.685                |
|  Perform Eigendecomposition   |  26.307               |  1                    |  26.307               |  0.47243              |
|  Save Covariance              |  11.445               |  1                    |  11.445               |  0.20552              |
|  Save Eigendecomposition      |  10.959               |  1                    |  10.959               |  0.1968               |
|  Save Lambda                  |  3.0458               |  1                    |  3.0458               |  0.054696             |
|  Save Pairwise Score          |  2.0978               |  1                    |  2.0978               |  0.037671             |
|  Load Covariance              |  0.72168              |  1                    |  0.72168              |  0.01296              |
|  Load Eigendecomposition      |  0.5194               |  1                    |  0.5194               |  0.0093274            |
|  Load All Factors             |  0.25427              |  1                    |  0.25427              |  0.0045661            |
----------------------------------------------------------------------------------------------------------------------------------
```

For faster computation, use half precision:

```bash
python analyze.py --dataset_name sst2 \
    --query_batch_size 175 \
    --train_batch_size 128 \
    --checkpoint_dir ./checkpoints \
    --factor_strategy ekfac \
    --use_half_precision
```

This reduces computation time to about 20 minutes on an A100 (80GB) GPU.

```
----------------------------------------------------------------------------------------------------------------------------------
|  Action                       |  Mean duration (s)    |  Num calls            |  Total time (s)       |  Percentage %         |
----------------------------------------------------------------------------------------------------------------------------------
|  Total                        |  -                    |  11                   |  3563.3               |  100 %                |
----------------------------------------------------------------------------------------------------------------------------------
|  Compute Pairwise Score       |  2877.1               |  1                    |  2877.1               |  80.742               |
|  Fit Lambda                   |  556.0                |  1                    |  556.0                |  15.603               |
|  Fit Covariance               |  99.458               |  1                    |  99.458               |  2.7912               |
|  Perform Eigendecomposition   |  15.968               |  1                    |  15.968               |  0.44812              |
|  Save Covariance              |  5.4501               |  1                    |  5.4501               |  0.15295              |
|  Save Eigendecomposition      |  5.3617               |  1                    |  5.3617               |  0.15047              |
|  Save Lambda                  |  1.533                |  1                    |  1.533                |  0.043022             |
|  Save Pairwise Score          |  1.123                |  1                    |  1.123                |  0.031517             |
|  Load Covariance              |  0.53788              |  1                    |  0.53788              |  0.015095             |
|  Load Eigendecomposition      |  0.52602              |  1                    |  0.52602              |  0.014762             |
|  Load All Factors             |  0.26329              |  1                    |  0.26329              |  0.0073888            |
----------------------------------------------------------------------------------------------------------------------------------
```

## Counterfactual Evaluation

Evaluate the impact of removing top positively influential training examples on query misclassification. 
First, compute pairwise influence scores for the `RTE` dataset:

```bash
python train.py --dataset_name rte \
    --checkpoint_dir ./checkpoints \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --learning_rate 2e-05 \
    --weight_decay 0.01 \
    --num_train_epochs 3 \
    --seed 0

python analyze.py --dataset_name rte \
    --query_batch_size 70 \
    --train_batch_size 128 \
    --checkpoint_dir ./checkpoints \
    --factor_strategy ekfac
    
python analyze.py --dataset_name rte \
    --query_batch_size 70 \
    --train_batch_size 128 \
    --checkpoint_dir ./checkpoints \
    --factor_strategy diagonal
```

Use `run_counterfactual.py` to run the counterfactual experiment.

<p align="center">
<a href="#"><img width="380" img src="figure/counterfactual.png" alt="Counterfactual"/></a>
</p>

## Evaluating Linear Datamodeling Score

The `evaluate_lds.py` script computes the [linear datamodeling score (LDS)](https://arxiv.org/abs/2303.14186). It measures the LDS obtained by 
retraining the network 500 times with different subsets of the dataset (5 repeats and 100 masks). 

<div align="center">

| Strategy                 | LDS	 |
|--------------------------|:----:|
| `identity`               | 0.10 |
| `diagonal`               | 0.15 |
| `kfac`                   | 0.32 |
| `ekfac`                  | 0.32 |
| `ekfac` (half precision) | 0.32 |

</div>

The script can also print top influential sequences for a given query.

```
Query Example:
 Sentence1: The west has preferred to focus on endangered animals, rather than endangered humans. African elephants are hunted down and stripped of tusks and hidden by poachers. Their numbers in Africa slumped from 1.2m to 600,000 in a decade until CITES - the Convention on International Trade in Endangered Species - banned the trade in ivory.
 Sentence2: African elephants are endangered by ivory poachers.
 Label: 0
 
Top Influential Example:
 Sentence1: The article also mentions the greater prevalence of obesity among two minority populations, African-Americans and Hispanic/Latino, but does not consider in its analysis of the increase in obesity the increase of these these populations as a proportion of the United States population.  African-Americans and Hispanic/Latinos have a higher rates of obesity than White Americans, while Asian-Americans have a relatively low rate of obesity. Despite only representing one third of the U.S. population, African-Americans and Hispanic/Latinos represent about one half of the population growth.
 Sentence2: African-Americans are a minority in the U.S.
 Label: 0
```