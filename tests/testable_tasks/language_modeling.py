# pylint: skip-file

from itertools import chain
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch import nn
from torch.utils import data
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Conv1D

from kronfluence.task import Task

BATCH_TYPE = Dict[str, torch.Tensor]


def _replace_conv1d_modules(model: nn.Module) -> None:
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            _replace_conv1d_modules(module)

        if isinstance(module, Conv1D):
            new_module = nn.Linear(in_features=module.weight.shape[0], out_features=module.weight.shape[1])
            new_module.weight.data.copy_(module.weight.data.t())
            new_module.bias.data.copy_(module.bias.data)
            setattr(model, name, new_module)


def make_tiny_gpt(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    config = AutoConfig.from_pretrained(
        "hf-internal-testing/tiny-random-gpt2",
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        "hf-internal-testing/tiny-random-gpt2",
        from_tf=False,
        config=config,
        ignore_mismatched_sizes=False,
        trust_remote_code=True,
    )
    _replace_conv1d_modules(model)
    return model


def make_gpt_dataset(num_data: int, seed: int = 0) -> data.Dataset:
    torch.manual_seed(seed)
    raw_datasets = load_dataset("wikitext", "wikitext-2-raw-v1")

    tokenizer = AutoTokenizer.from_pretrained(
        "hf-internal-testing/tiny-random-gpt2", use_fast=True, trust_remote_code=True
    )
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=None,
        remove_columns=column_names,
        load_from_cache_file=True,
    )

    block_size = 32

    def group_texts(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=None,
        load_from_cache_file=True,
    )

    train_dataset = lm_datasets["train"]
    ds = train_dataset
    ds = ds.select(range(num_data))
    return ds


class LanguageModelingTask(Task):
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

        shift_logits = logits[..., :-1, :].contiguous()

        if not sample:
            labels = batch["labels"]
            shift_labels = labels[..., 1:].contiguous()
            reshaped_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            summed_loss = F.cross_entropy(reshaped_shift_logits, shift_labels.view(-1), reduction="sum")
        else:
            reshaped_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            with torch.no_grad():
                probs = torch.nn.functional.softmax(reshaped_shift_logits, dim=-1)
                sampled_labels = torch.multinomial(
                    probs,
                    num_samples=1,
                ).flatten()
            summed_loss = F.cross_entropy(reshaped_shift_logits, sampled_labels.detach(), reduction="sum")
        return summed_loss

    def compute_measurement(
        self,
        batch: BATCH_TYPE,
        model: nn.Module,
    ) -> torch.Tensor:
        return self.compute_train_loss(batch, model)

    def tracked_modules(self) -> List[str]:
        total_modules = []

        for i in range(5):
            total_modules.append(f"transformer.h.{i}.attn.c_attn")
            total_modules.append(f"transformer.h.{i}.attn.c_proj")

        for i in range(5):
            total_modules.append(f"transformer.h.{i}.mlp.c_fc")
            total_modules.append(f"transformer.h.{i}.mlp.c_proj")

        return total_modules

    def get_attention_mask(self, batch: Any) -> Optional[torch.Tensor]:
        return batch["attention_mask"]
