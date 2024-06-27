# SWAG & RoBERTa Example

This directory contains scripts for fine-tuning RoBERTa computing influence scores on the SWAG dataset. The pipeline is motivated from [this HuggingFace Example](https://github.com/huggingface/transformers/tree/main/examples/pytorch/multiple-choice).
To get started, please install the necessary packages:

```bash
pip install -r requirements.txt
```

## Training

To fine-tune RoBERTa on SWAG, run the following command:

```bash
python train.py --checkpoint_dir ./checkpoints \
    --train_batch_size 16 \
    --eval_batch_size 32 \
    --learning_rate 1e-05 \
    --weight_decay 0.001 \
    --num_train_epochs 3 \
    --seed 1004
```

This will fine-tune the model using the specified hyperparameters and save the final checkpoint in the `./checkpoints` directory.

## Computing Pairwise Influence Scores

To obtain pairwise influence scores on 2000 query data points using `ekfac`, run the following command:

```bash
python analyze.py --query_batch_size 64 \
    --train_batch_size 128 \
    --use_half_precision \
    --checkpoint_dir ./checkpoints \
    --factor_strategy ekfac
```

You can also use `identity`, `diagonal`, and `kfac` for `factor_strategy`. On an A6000 (48GB), it takes roughly 95 minutes to compute the pairwise scores (including computing EKFAC factors):

```

```

For more efficient computation, use AMP half precision + query batching + DDP:

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=2 analyze.py --factor_batch_size 128 \
    --query_batch_size 100 \
    --train_batch_size 128 \
    --checkpoint_dir ./checkpoints \
    --factor_strategy ekfac \
    --query_gradient_rank 32 \
    --use_half_precision \
    --use_ddp
```

This reduces computation time to about 20 minutes on an A100 (80GB) GPU:

```

```

## Evaluating Linear Datamodeling Score

The `evaluate_lds.py` script computes the [linear datamodeling score (LDS)](https://arxiv.org/abs/2303.14186). It measures the LDS obtained by 
retraining the network 500 times with different subsets of the dataset (5 repeats and 100 masks). 
We obtain `xx` LDS (we get `xx` LDS with the AMP half precision + query batching + DDP).

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
