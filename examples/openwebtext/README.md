# OpenWebText & Llama-3-8B Example

This repository contains scripts for computing influence scores on the subset of OpenWebText dataset. The pipeline is motivated from [LoggIX repository](https://github.com/logix-project/logix/tree/main/examples/language_modeling).
Install the necessary packages:

```bash
pip install -r requirements.txt
```

We will use the pre-trained Meta-Llama-3-8B model [from HuggingFace](https://huggingface.co/meta-llama/Meta-Llama-3-8B).

## Computing EKFAC Factors

To compute factors using the `ekfac` factorization strategy, run the following command which uses 4 A100 (80GB) GPUs:

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=4 fit_factors.py --factor_batch_size 4
```

## Computing Influence Scores

The `generate.py` folder contains a code to generate response of the Llama-3-8B model given certain prompt.
I saved some prompt and completition pair to the directory `data/data.json`. 

To compute influence scores on the generated prompt and compleition pair, run the following command:

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=4 compute_scores.py --train_batch_size 8
```