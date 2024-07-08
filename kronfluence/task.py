from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import torch
from torch import nn


class Task(ABC):
    """Abstract base class for task definitions.

    Extend this class to implement specific tasks (e.g., regression, classification, language modeling)
    with custom pipelines (e.g., models, data loaders, training objectives).

    Attributes:
        enable_post_process_per_sample_gradient (bool):
            Flag to enable post-processing of per-sample gradients. Defaults to `False`.
    """

    enable_post_process_per_sample_gradient: bool = False

    @abstractmethod
    def compute_train_loss(
        self,
        batch: Any,
        model: nn.Module,
        sample: bool = False,
    ) -> torch.Tensor:
        """Computes the training loss for a given batch and model.

        Args:
            batch (Any):
                A batch of data from the DataLoader.
            model (nn.Module):
                The PyTorch model used for loss computation.
            sample (bool):
                Indicates whether to sample from the model's outputs or to use the actual targets from the
                batch. Defaults to `False`. The case where `sample=True` must be implemented to
                approximate the true Fisher.

        Returns:
            torch.Tensor:
                The computed loss as a scalar tensor.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement the `compute_train_loss` method.")

    @abstractmethod
    def compute_measurement(
        self,
        batch: Any,
        model: nn.Module,
    ) -> torch.Tensor:
        """Computes a measurable quantity for a given batch and model.

        This method calculates f(θ) as defined in https://arxiv.org/pdf/2308.03296.pdf. The measurable quantity
        can be a loss, logit, log probability, or any other relevant metric for the task.

        Args:
            batch (Any):
                A batch of data from the DataLoader.
            model (nn.Module):
                The PyTorch model used for measurement computation.

        Returns:
            torch.Tensor:
                The computed measurable quantity as a tensor.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement the `compute_measurement` method.")

    def get_influence_tracked_modules(self) -> Optional[List[str]]:
        """Specifies which modules should be tracked for influence factor and score computations.

        Override this method in subclasses to return a list of specific module names if influence functions
        should only be computed for a subset of the model.

        Returns:
            Optional[List[str]]:
                A list of module names to compute influence functions for, or `None` to compute for
                all applicable modules (e.g., `nn.Linear` and `nn.Conv2d`).
        """

    def get_attention_mask(self, batch: Any) -> Optional[Union[Dict[str, torch.Tensor], torch.Tensor]]:
        """Gets attention masks for padded sequences in a batch.

        This method is typically used for models or datasets that require masking, such as transformer-based
        architectures. For more information, see: https://huggingface.co/docs/transformers/en/glossary#attention-mask.

        Args:
            batch (Any):
                A batch of data from the DataLoader.

        Returns:
            Optional[Union[Dict[str, torch.Tensor], torch.Tensor]]:
                - `None` if padding is not used.
                - A binary tensor with dimension `batch_size x num_seq` as the mask for the batch.
                - A dictionary mapping module names to their corresponding masks for models requiring different
                masks for different modules (e.g., encoder-decoder architectures).
        """

    def post_process_per_sample_gradient(self, module_name: str, gradient: torch.Tensor) -> torch.Tensor:
        """Post-processes the per-sample gradient of a specific module.

        This method is called only if `do_post_process_per_sample_gradient` is set to `True`.
        Override this method in subclasses to implement custom gradient post-processing.

        Args:
            module_name (str):
                The name of the module whose gradient is being processed.
            gradient (torch.Tensor):
                The per-sample gradient tensor with dimension `batch_size x gradient_dim x activation_dim`.

        Returns:
            torch.Tensor:
                The modified per-sample gradient tensor.
        """
        del module_name
        return gradient
