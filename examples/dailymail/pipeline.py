from typing import Any, List

import torch.nn as nn
from datasets import Dataset, load_dataset
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

summarization_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
    "multi_news": ("document", "summary"),
}


MODEL_NAME = "google-t5/t5-small"


def construct_t5() -> nn.Module:
    config = AutoConfig.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )
    return AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        from_tf=False,
        config=config,
        ignore_mismatched_sizes=False,
        trust_remote_code=True,
    )


def get_tokenizer() -> Any:
    return AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, trust_remote_code=True)


def get_dailymail_dataset(
    split: str,
    indices: List[int] = None,
) -> Dataset:
    raw_datasets = load_dataset("cnn_dailymail", "3.0.0")

    tokenizer = get_tokenizer()
    column_names = raw_datasets["train"].column_names
    dataset_columns = summarization_name_mapping.get("cnn_dailymail", None)
    text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]

    max_source_length = 1024
    max_target_length = 128
    padding = False
    prefix = "summarize: "

    def preprocess_function(examples):
        inputs = examples[text_column]
        targets = examples[summary_column]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

        labels = tokenizer(
            text_target=targets,
            max_length=max_target_length,
            padding=padding,
            truncation=True,
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    if split == "train" or split == "eval_train":
        train_dataset = raw_datasets["train"]
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=None,
            remove_columns=column_names,
            load_from_cache_file=True,
            desc="Running tokenizer on dataset.",
        )
        ds = train_dataset
    else:
        valid_dataset = raw_datasets["validation"]
        eval_dataset = valid_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=None,
            remove_columns=column_names,
            load_from_cache_file=True,
            desc="Running tokenizer on dataset.",
        )
        ds = eval_dataset

    if indices is not None:
        ds = ds.select(indices)

    return ds


if __name__ == "__main__":
    from kronfluence import Analyzer

    model = construct_t5()
    print(Analyzer.get_module_summary(model))
