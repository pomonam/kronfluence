# pylint: skip-file

from itertools import chain
from typing import Any, Dict

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch import nn
from torch.utils import data
from transformers import AutoConfig, AutoModelForMultipleChoice, AutoTokenizer, logging

from kronfluence.task import Task

logging.set_verbosity_error()
BATCH_TYPE = Dict[str, torch.Tensor]


def make_tiny_roberta(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    config = AutoConfig.from_pretrained(
        "hf-internal-testing/tiny-random-RobertaModel",
        trust_remote_code=True,
    )
    model = AutoModelForMultipleChoice.from_pretrained(
        "hf-internal-testing/tiny-random-RobertaModel",
        from_tf=False,
        config=config,
        ignore_mismatched_sizes=False,
        trust_remote_code=True,
    )
    return model


def make_roberta_dataset(num_data: int, seed: int = 0) -> data.Dataset:
    torch.manual_seed(seed)
    raw_datasets = load_dataset("swag", "regular")
    tokenizer = AutoTokenizer.from_pretrained(
        "hf-internal-testing/tiny-random-RobertaModel", use_fast=True, trust_remote_code=True
    )

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

        first_sentences = list(chain(*first_sentences))
        second_sentences = list(chain(*second_sentences))

        tokenized_examples = tokenizer(
            first_sentences,
            second_sentences,
            max_length=128,
            padding=padding,
            truncation=True,
        )
        tokenized_inputs = {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    processed_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )

    dataset = processed_datasets["train"].select(range(num_data))

    return dataset


class MultipleChoiceTask(Task):
    enable_post_process_per_sample_gradient = True

    def compute_train_loss(
        self,
        batch: BATCH_TYPE,
        model: nn.Module,
        sample: bool = False,
    ) -> torch.Tensor:
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).logits

        if not sample:
            return F.cross_entropy(logits, batch["labels"], reduction="sum")
        with torch.no_grad():
            probs = torch.nn.functional.softmax(logits.detach(), dim=-1)
            sampled_labels = torch.multinomial(
                probs,
                num_samples=1,
            ).flatten()
        return F.cross_entropy(logits, sampled_labels, reduction="sum")

    def compute_measurement(
        self,
        batch: BATCH_TYPE,
        model: nn.Module,
    ) -> torch.Tensor:
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).logits

        labels = batch["labels"]
        bindex = torch.arange(logits.shape[0]).to(device=logits.device, non_blocking=False)
        logits_correct = logits[bindex, labels]

        cloned_logits = logits.clone()
        cloned_logits[bindex, labels] = torch.tensor(-torch.inf, device=logits.device, dtype=logits.dtype)

        margins = logits_correct - cloned_logits.logsumexp(dim=-1)
        return -margins.sum()

    def get_attention_mask(self, batch: Any) -> torch.Tensor:
        return batch["attention_mask"]

    def post_process_per_sample_gradient(self, module_name: str, gradient: torch.Tensor) -> torch.Tensor:
        del module_name
        total_batch_size = gradient.size(0)
        true_batch_size = int(total_batch_size / 4)
        return gradient.reshape(true_batch_size, 4, *gradient.size()[1:]).sum(dim=1)
