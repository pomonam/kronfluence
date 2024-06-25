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

To obtain pairwise influence scores on 2000 query data points using `ekfac`, run the following command:
```bash
python analyze.py --query_batch_size 1000 \
    --dataset_dir ./data \
    --checkpoint_dir ./checkpoints \
    --factor_strategy ekfac
```
You can also use `identity`, `diagonal`, and `kfac`. On A100 (80GB), it takes roughly 1.5 minutes to compute the 
pairwise scores (including computing EKFAC factors).

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