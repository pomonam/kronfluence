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
|  Total                        |  -                    |  11                   |  7330.2               |  100 %                |
----------------------------------------------------------------------------------------------------------------------------------
|  Compute Pairwise Score       |  4424.5               |  1                    |  4424.5               |  60.359               |
|  Fit Lambda                   |  2366.2               |  1                    |  2366.2               |  32.28                |
|  Fit Covariance               |  483.92               |  1                    |  483.92               |  6.6017               |
|  Perform Eigendecomposition   |  26.227               |  1                    |  26.227               |  0.35779              |
|  Save Covariance              |  11.824               |  1                    |  11.824               |  0.1613               |
|  Save Eigendecomposition      |  11.027               |  1                    |  11.027               |  0.15043              |
|  Save Lambda                  |  3.1113               |  1                    |  3.1113               |  0.042445             |
|  Save Pairwise Score          |  2.0391               |  1                    |  2.0391               |  0.027818             |
|  Load Covariance              |  0.56135              |  1                    |  0.56135              |  0.007658             |
|  Load Eigendecomposition      |  0.55821              |  1                    |  0.55821              |  0.0076152            |
|  Load All Factors             |  0.29442              |  1                    |  0.29442              |  0.0040165            |
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