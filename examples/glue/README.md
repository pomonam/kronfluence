# GLUE & BERT Example

This directory contains scripts for fine-tuning BERT on GLUE benchmark. The pipeline is motivated from [HuggingFace Example](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification).
Please begin by installing necessary packages.
```bash
pip install -r requirements.txt
```

## Training

To fine-tune BERT on some specific dataset, run the following command (we are using `SST2` dataset):
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

## Computing Pairwise Influence Scores

To obtain a pairwise influence scores on maximum of 2000 query data points using `ekfac`, run the following command:
```bash
python analyze.py --dataset_name sst2 \
    --query_batch_size 200 \
    --train_batch_size 256 \
    --checkpoint_dir ./checkpoints \
    --factor_strategy ekfac
```
On A100 (80GB), it takes roughly 100 minutes to compute the pairwise scores (including computing EKFAC factors).
