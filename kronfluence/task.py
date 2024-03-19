from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import torch
from torch import nn


class Task(ABC):
    """Abstract base class for task definitions.

    Extend this class to implement specific tasks (e.g., regression, classification, language modeling)
    with custom pipelines (models, data loaders, training objectives).
    """

    @abstractmethod
    def compute_train_loss(
        self,
        batch: Any,
        model: nn.Module,
        sample: bool = False,
    ) -> torch.Tensor:
        """Computes training loss for a given batch and model.

        Args:
            batch (Any):
                Batch of data sourced from the DataLoader.
            model (nn.Module):
                The PyTorch model for loss computation.
            sample (bool):
                Indicates whether to sample from the model's outputs or to use the actual targets from the
                batch. Defaults to False. The case where `sample` is set to True must be implemented to
                approximate the true Fisher.

        Returns:
            torch.Tensor:
                The computed loss as a tensor.
        """
        raise NotImplementedError("Subclasses must implement the `compute_train_loss` method.")

    @abstractmethod
    def compute_measurement(
        self,
        batch: Any,
        model: nn.Module,
    ) -> torch.Tensor:
        """Computes a measurable quantity (e.g., loss, logit, log probability) for a given batch and model.
        This is defined as f(θ) from https://arxiv.org/pdf/2308.03296.pdf.

        Args:
            batch (Any):
                Batch of data sourced from the DataLoader.
            model (nn.Module):
                The PyTorch model for measurement computation.

        Returns:
            torch.Tensor:
                The measurable quantity as a tensor.
        """
        raise NotImplementedError("Subclasses must implement the `compute_measurement` method.")

    def tracked_modules(self) -> Optional[List[str]]:
        """Specifies modules for influence score computations.

        Returns None by default, applying computations to all supported modules (e.g., nn.Linear, nn.Conv2d).
        Subclasses can override this method to return a list of specific module names if influence functions
        should only be computed for a subset of the model.

        Returns:
            Optional[List[str]]:
                A list of module names for which to compute influence functions, or None to indicate that
                influence functions should be computed for all applicable modules.
        """

    def get_attention_mask(self, batch: Any) -> Optional[Union[Dict[str, torch.Tensor], torch.Tensor]]:
        """Returns masks for data points within a batch that have been padded extra tokens to ensure
        consistent length across the batch. Typically, it returns None for models or datasets not requiring
        masking.

        See https://huggingface.co/docs/transformers/en/glossary#attention-mask.

        Args:
            batch (Any):
                Batch of data sourced from the DataLoader.

        Returns:
            Optional[Union[Dict[str, torch.Tensor], torch.Tensor]]:
                A binary tensor as the mask for the batch, or None if padding is not used. The mask dimensions should
                match `batch_size x num_seq`. For models requiring different masks for various modules
                (e.g., encoder-decoder architectures), returns a dictionary mapping module names to their
                corresponding masks.
        """
