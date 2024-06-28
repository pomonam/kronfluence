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
    --train_batch_size 64 \
    --use_half_precision \
    --checkpoint_dir ./checkpoints \
    --factor_strategy ekfac
```

You can also use `identity`, `diagonal`, and `kfac` for `factor_strategy`. On an A100 (80GB), it takes roughly 8 hours to compute the pairwise scores (including computing EKFAC factors).

For more efficient computation, use DDP:

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=2 analyze.py --factor_batch_size 128 \
    -train_batch_size 64 \
    --use_half_precision \
    --checkpoint_dir ./checkpoints \
    --factor_strategy ekfac \
    --use_ddp
```

## Evaluating Linear Datamodeling Score

The `evaluate_lds.py` script computes the [linear datamodeling score (LDS)](https://arxiv.org/abs/2303.14186). It measures the LDS obtained by 
retraining the network 500 times with different subsets of the dataset (5 repeats and 100 masks). We obtain `xx` LDS.
