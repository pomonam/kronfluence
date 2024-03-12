# UCI Regression Example

This directory contains scripts designed for training a regression model and conducting influence analysis with datasets obtained from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/datasets).

## Training

To initiate the training of a regression model using the Concrete dataset, execute the following command:
```bash
python train.py --dataset_name concrete \ 
    --dataset_dir ./data \
    --output_dir ./checkpoints \
    --train_batch_size 32 \
    --eval_batch_size 1024 \
    --learning_rate 0.03 \
    --weight_decay 1e-5 \
    --num_train_epochs 20 \
    --seed 1004
```
Alternatively, you can download the model checkpoint.

# Influence Analysis

To obtain a pairwise influence scores using EKFAC, 

# Counterfactual Evaluation

To evaluate the accuracy of influence estimates, we can perform counterfactual evaluation.