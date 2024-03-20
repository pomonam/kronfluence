from typing import List

from datasets import load_dataset
from torch import nn
from torch.utils.data import Dataset
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

# Copied from https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py.
GLUE_TASK_TO_KEYS = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def construct_bert(data_name: str = "sst2") -> nn.Module:
    config = AutoConfig.from_pretrained(
        "bert-base-cased",
        num_labels=2,
        finetuning_task=data_name,
        trust_remote_code=True,
    )
    return AutoModelForSequenceClassification.from_pretrained(
        "bert-base-cased",
        from_tf=False,
        config=config,
        ignore_mismatched_sizes=False,
        trust_remote_code=True,
    )


def get_glue_dataset(
    data_name: str,
    split: str,
    indices: List[int] = None,
) -> Dataset:
    assert split in ["train", "eval_train", "valid"]

    raw_datasets = load_dataset(
        path="glue",
        name=data_name,
    )
    label_list = raw_datasets["train"].features["label"].names
    num_labels = len(label_list)
    assert num_labels == 2

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", use_fast=True, trust_remote_code=True)

    sentence1_key, sentence2_key = GLUE_TASK_TO_KEYS[data_name]
    padding = "max_length"
    max_seq_length = 128

    def preprocess_function(examples):
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=max_seq_length, truncation=True)
        if "label" in examples:
            result["labels"] = examples["label"]
        return result

    raw_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        load_from_cache_file=(not False),
    )

    if split in ["train", "eval_train"]:
        train_dataset = raw_datasets["train"]
        ds = train_dataset
    else:
        eval_dataset = raw_datasets["validation"]
        ds = eval_dataset

    if indices is not None:
        ds = ds.select(indices)

    return ds


if __name__ == "__main__":
    from kronfluence import Analyzer

    model = construct_bert()
    print(Analyzer.get_module_summary(model))
