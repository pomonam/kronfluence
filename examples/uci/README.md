# UCI Regression Example

This directory contains scripts for training a regression model and conducting influence analysis with 
datasets from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/datasets). Please begin by installing necessary packages.
```bash
pip install -r requirements.txt
```

## Training

To train a regression model on the `Concrete` dataset, run the following command:
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

## Computing Pairwise Influence Scores

To obtain a pairwise influence scores using `ekfac`, run the following command:
```bash
python analyze.py --dataset_name concrete \
    --dataset_dir ./data \
    --checkpoint_dir ./checkpoints \
    --factor_strategy ekfac
```
You can also use `identity`, `diagonal`, and `kfac`.

## Counterfactual Evaluation

You can check the notebook `tutorial.ipynb` to run the subset removal counterfactual evaluation. 
(Note that `TracIn` uses the final checkpoint instead of the intermediate checkpoints throughout training.)

<p align="center">
<a href="#"><img width="380" img src="figure/counterfactual.png" alt="Counterfactual"/></a>
</p>
