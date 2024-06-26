# SWAG & RoBERTa Example

This directory contains scripts for fine-tuning RoBERTa computing influence scores on SWAG dataset. The pipeline is motivated from [this HuggingFace Example](https://github.com/huggingface/transformers/tree/main/examples/pytorch/multiple-choice).
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

## Computing Pairwise Influence Scores

To obtain pairwise influence scores on a maximum of 2000 query data points using `ekfac`, run the following command:

```bash
python analyze.py --dataset_name sst2 \
    --query_batch_size 175 \
    --train_batch_size 128 \
    --checkpoint_dir ./checkpoints \
    --factor_strategy ekfac
```

On an A100 (80GB), it takes roughly 95 minutes to compute the pairwise scores for SST2 with around 900 query data points (including computing EKFAC factors):

```

```

For more efficient computation, use half precision:

```bash
python analyze.py --dataset_name sst2 \
    --query_batch_size 175 \
    --train_batch_size 128 \
    --checkpoint_dir ./checkpoints \
    --factor_strategy ekfac \
    --use_half_precision
```

This reduces computation time to about 20 minutes on an A100 (80GB) GPU:

```

```

## Counterfactual Evaluation

Can we remove top positively influential training examples to make some queries misclassify?
Subset removal counterfactual evaluation selects correctly classified query data point, removes 
top-k positively influential training samples, and retrain the network with the modified dataset to see if that query 
data point gets misclassified. 

<p align="center">
<a href="#"><img width="380" img src="figure/counterfactual.png" alt="Counterfactual"/></a>
</p>

