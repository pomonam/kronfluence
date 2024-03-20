# pylint: skip-file

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch import nn
from torch.utils import data
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from kronfluence.task import Task

BATCH_TYPE = Dict[str, torch.Tensor]


def make_tiny_bert(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    config = AutoConfig.from_pretrained(
        "hf-internal-testing/tiny-bert",
        num_labels=2,
        finetuning_task="rte",
        trust_remote_code=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        "hf-internal-testing/tiny-bert",
        from_tf=False,
        config=config,
        ignore_mismatched_sizes=False,
        trust_remote_code=True,
    )
    return model


def make_bert_dataset(num_data: int, do_not_pad: bool = False, seed: int = 0) -> data.Dataset:
    torch.manual_seed(seed)
    raw_datasets = load_dataset(
        "glue",
        "rte",
    )
    label_list = raw_datasets["train"].features["label"].names
    num_labels = len(label_list)
    assert num_labels == 2

    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-bert", use_fast=True, trust_remote_code=True)
    sentence1_key, sentence2_key = ("sentence1", "sentence2")
    padding = "max_length"
    max_seq_length = 128
    if do_not_pad:
        padding = "do_not_pad"

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
    train_dataset = raw_datasets["train"]
    ds = train_dataset
    ds = ds.select(range(num_data))
    return ds


class TextClassificationTask(Task):
    def compute_train_loss(
        self,
        batch: BATCH_TYPE,
        model: nn.Module,
        sample: bool = False,
    ) -> torch.Tensor:
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
        ).logits

        if not sample:
            return F.cross_entropy(logits, batch["labels"], reduction="sum")
        with torch.no_grad():
            probs = torch.nn.functional.softmax(logits, dim=-1)
            sampled_labels = torch.multinomial(
                probs,
                num_samples=1,
            ).flatten()
        return F.cross_entropy(logits, sampled_labels.detach(), reduction="sum")

    def compute_measurement(
        self,
        batch: BATCH_TYPE,
        model: nn.Module,
    ) -> torch.Tensor:
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
        ).logits

        labels = batch["labels"]
        bindex = torch.arange(logits.shape[0]).to(device=logits.device, non_blocking=False)
        logits_correct = logits[bindex, labels]

        cloned_logits = logits.clone()
        cloned_logits[bindex, labels] = torch.tensor(-torch.inf, device=logits.device, dtype=logits.dtype)

        margins = logits_correct - cloned_logits.logsumexp(dim=-1)
        return -margins.sum()

    def get_attention_mask(self, batch: Any) -> Optional[torch.Tensor]:
        return batch["attention_mask"]
