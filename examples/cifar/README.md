# CIFAR-10 & ResNet-9 Example

This directory contains scripts designed for training ResNet-9 on CIFAR-10. The pipeline is motivated from 
[TRAK repository](https://github.com/MadryLab/trak/blob/main/examples/cifar_quickstart.ipynb).

## Training

To train a regression model on the Concrete dataset, run the following command:
```bash
python train.py --dataset_dir ./data --checkpoint_dir ./checkpoints \
    --train_batch_size 512 \
    --eval_batch_size 1024 \
    --learning_rate 0.4 \
    --weight_decay 0.001 \
    --num_train_epochs 25 \
    --seed 1004
```

# Computing Pairwise Influence Scores

To obtain a pairwise influence scores using EKFAC, run the following command:
```bash
python analyze.py --dataset_name concrete \ 
    --dataset_dir ./data \
    --checkpoint_dir ./checkpoints \
    --factor_strategy ekfac
```

# Counterfactual Evaluation

You can check the notebook `tutorial.ipynb` for running the counterfactual evaluation.