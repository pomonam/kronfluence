from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import nn

from kronfluence.task import Task

BATCH_TYPE = Dict[str, torch.Tensor]


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
        logits = logits[..., :-1, :].contiguous()
        logits = logits.view(-1, logits.size(-1))

        if not sample:
            labels = batch["labels"]
            shift_labels = labels[..., 1:].contiguous()
            summed_loss = F.cross_entropy(logits, shift_labels.view(-1), reduction="sum", ignore_index=-100)
        else:
            with torch.no_grad():
                probs = torch.nn.functional.softmax(logits.detach(), dim=-1)
                sampled_labels = torch.multinomial(
                    probs,
                    num_samples=1,
                ).flatten()
            summed_loss = F.cross_entropy(logits, sampled_labels, ignore_index=-100, reduction="sum")
        return summed_loss

    def compute_measurement(
        self,
        batch: BATCH_TYPE,
        model: nn.Module,
    ) -> torch.Tensor:
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).logits
        shift_labels = batch["labels"][..., 1:].contiguous().view(-1)
        logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
        return F.cross_entropy(logits, shift_labels, ignore_index=-100, reduction="sum")

    def get_influence_tracked_modules(self) -> List[str]:
        total_modules = []

        # You can uncomment the following lines if you would like to compute influence also on attention layers.
        # for i in range(32):
        #     total_modules.append(f"model.layers.{i}.self_attn.q_proj")
        #     total_modules.append(f"model.layers.{i}.self_attn.k_proj")
        #     total_modules.append(f"model.layers.{i}.self_attn.v_proj")
        #     total_modules.append(f"model.layers.{i}.self_attn.o_proj")

        for i in range(32):
            total_modules.append(f"model.layers.{i}.mlp.gate_proj")
            total_modules.append(f"model.layers.{i}.mlp.up_proj")
            total_modules.append(f"model.layers.{i}.mlp.down_proj")

        return total_modules

    def get_attention_mask(self, batch: BATCH_TYPE) -> torch.Tensor:
        return batch["attention_mask"]
