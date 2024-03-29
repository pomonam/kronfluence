# WikiText & GPT-2 Example

This directory contains scripts for fine-tuning GPT-2 on WikiText2 dataset. The pipeline is motivated from 
[HuggingFace Example](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling).
Please begin by installing necessary packages.
```bash
pip install -r requirements.txt
```

## Training

To fine-tune GPT-2, run the following command:
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
You can also use `identity`, `diagonal`, and `kfac`. On A100 (80GB), it takes roughly 50 minutes to compute the 
pairwise scores (including computing EKFAC factors).


## Counterfactual Experiment

We can conduct counterfactual experiment by observing the increase in validation perplexity when removing top influential sequences.
We show a simple demo in `run_counterfactual.py` (the code assumes that you have computed the pairwise influence scores with `ekfac` and `identity`).
<p align="center">
<a href="#"><img width="380" img src="figure/counterfactual.png" alt="Counterfactual"/></a>
</p>



## Computing Linear Datamodeling Score

We can also compute the [Linear Datamodeling Score (LDS)](https://arxiv.org/abs/2303.14186). The code in `evaluate_lds.py` measures the LDS obtained by 
retraining the network 600 times with different subsets of the dataset (5 repeats and 120 masks). We can obtain `0.37` LDS.