from dataclasses import dataclass
from itertools import chain
from typing import Any, List, Optional, Union

import torch
from datasets import load_dataset
from torch import nn
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)
from transformers.utils import PaddingStrategy


def construct_roberta() -> nn.Module:
    config = AutoConfig.from_pretrained(
        "FacebookAI/roberta-base",
        trust_remote_code=True,
    )
    return AutoModelForMultipleChoice.from_pretrained(
        "FacebookAI/roberta-base",
        config=config,
        trust_remote_code=True,
    )


@dataclass
class DataCollatorForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = list(chain(*flattened_features))

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


def get_swag_dataset(
    split: str,
    indices: List[int] = None,
) -> Dataset:
    assert split in ["train", "eval_train", "valid"]

    raw_datasets = load_dataset("swag", "regular")

    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base", use_fast=True, trust_remote_code=True)

    column_names = raw_datasets["train"].column_names
    ending_names = [f"ending{i}" for i in range(4)]
    context_name = "sent1"
    question_header_name = "sent2"
    label_column_name = "label" if "label" in column_names else "labels"
    padding = "max_length"

    def preprocess_function(examples: Any):
        first_sentences = [[context] * 4 for context in examples[context_name]]
        question_headers = examples[question_header_name]
        second_sentences = [
            [f"{header} {examples[end][i]}" for end in ending_names] for i, header in enumerate(question_headers)
        ]
        labels = examples[label_column_name]

        # Flatten out.
        first_sentences = list(chain(*first_sentences))
        second_sentences = list(chain(*second_sentences))

        # Tokenize.
        tokenized_examples = tokenizer(
            first_sentences,
            second_sentences,
            max_length=128,
            padding=padding,
            truncation=True,
        )
        # Un-flatten.
        tokenized_inputs = {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    processed_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )

    if split in ["train", "eval_train"]:
        dataset = processed_datasets["train"]
        dataset = dataset.select(list(range(73_536)))
    else:
        dataset = processed_datasets["validation"]
        dataset = dataset.select(list(range(2000)))

    if indices is not None:
        dataset = dataset.select(indices)

    return dataset


if __name__ == "__main__":
    from kronfluence import Analyzer

    model = construct_roberta()
    print(Analyzer.get_module_summary(model))
