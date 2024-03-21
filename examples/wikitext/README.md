# WikiText & GPT-2 Example

This directory contains scripts for fine-tuning BERT on GLUE benchmark. The pipeline is motivated from 
[HuggingFace Example](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling).
Please begin by installing necessary packages.
```bash
pip install -r requirements.txt
```

## Training

To fine-tune BERT on some specific dataset, run the following command (we are using `SST2` dataset):
```bash
python train.py --checkpoint_dir ./checkpoints \
    --train_batch_size 8 \
    --eval_batch_size 16 \
    --learning_rate 3e-05 \
    --weight_decay 0.01 \
    --num_train_epochs 3 
```

## Computing Pairwise Influence Scores

To obtain a pairwise influence scores on 481 query data points using `ekfac`, run the following command:
```bash
python analyze.py --query_batch_size 32 \
    --train_batch_size 64 \
    --checkpoint_dir ./checkpoints \
    --factor_strategy ekfac
```
You can also use `identity`, `diagonal`, and `kfac`. On A100 (80GB), it takes roughly 1.5 minutes to compute the 
pairwise scores (including computing EKFAC factors).

Can we speed up the computations?
```bash
python analyze.py --query_batch_size 97 \
    --train_batch_size 64 \
    --query_gradient_rank 32 \
    --use_half_precision true \
    --checkpoint_dir ./checkpoints \
    --factor_strategy ekfac
```

## Computing Linear Datamodeling Score

We can compute the [Linear Datamodeling Score (LDS)](https://arxiv.org/abs/2303.14186). The code in `evaluate_lds.py`
measures the LDS obtained by retraining the network 600 times with different subsets of the dataset (5 repeats and 120 masks).