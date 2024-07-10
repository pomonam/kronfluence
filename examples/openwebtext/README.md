# OpenWebText & Llama-3-8B Example

This repository contains scripts for computing influence scores on the subset of OpenWebText dataset. 
The pipeline is inspired by [the LoggIX](https://github.com/logix-project/logix/tree/main/examples/language_modeling).
Install the necessary packages:

```bash
pip install -r requirements.txt
```

## Training

We will use the pre-trained model [from HuggingFace](https://huggingface.co/meta-llama/Meta-Llama-3-8B).

## Computing EKFAC Factors

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=4 fit_factors.py --factor_batch_size 4
```


The `generate.py` folder contains a code to generate response of the Llama-3-8B model given certain prompt.
I saved some prompt and completition pair to the directory `data/data.json`. 