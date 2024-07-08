# pylint: skip-file

import copy

import pytest
import torch
from torch.utils.data import DataLoader

from kronfluence.arguments import FactorArguments
from kronfluence.module.tracked_module import ModuleMode
from kronfluence.module.utils import set_mode, wrap_tracked_modules
from tests.utils import prepare_test


@pytest.mark.parametrize(
    "test_name",
    ["mlp", "conv_bn", "gpt"],
)
@pytest.mark.parametrize(
    "mode",
    [
        ModuleMode.DEFAULT,
        ModuleMode.COVARIANCE,
        ModuleMode.LAMBDA,
        ModuleMode.PRECONDITION_GRADIENT,
        ModuleMode.GRADIENT_AGGREGATION,
    ],
)
@pytest.mark.parametrize("train_size", [32])
@pytest.mark.parametrize("seed", [0])
def test_tracked_modules_forward_equivalence(
    test_name: str,
    mode: ModuleMode,
    train_size: int,
    seed: int,
) -> None:
    # The forward pass should produce the same results with and without wrapped modules.
    model, train_dataset, _, data_collator, task = prepare_test(
        test_name=test_name,
        train_size=train_size,
        seed=seed,
    )
    batch_size = 4
    train_loader = DataLoader(
        train_dataset,
        collate_fn=data_collator,
        batch_size=batch_size,
        drop_last=False,
        shuffle=False,
    )
    factor_args = FactorArguments(
        strategy="identity",
    )
    wrapped_model = wrap_tracked_modules(copy.deepcopy(model), factor_args=factor_args)

    original_losses = []
    for batch in train_loader:
        model.zero_grad(set_to_none=True)
        original_loss = task.compute_train_loss(batch, model, sample=False)
        original_loss.backward()
        original_losses.append(original_loss.detach())

    set_mode(model=wrapped_model, mode=mode)
    wrapped_losses = []
    for batch in train_loader:
        wrapped_model.zero_grad(set_to_none=True)
        wrapped_loss = task.compute_train_loss(batch, wrapped_model, sample=False)
        wrapped_loss.backward()
        wrapped_losses.append(wrapped_loss.detach())

    for i in range(len(original_losses)):
        assert torch.allclose(original_losses[i], wrapped_losses[i])


@pytest.mark.parametrize(
    "test_name",
    ["mlp", "conv"],
)
@pytest.mark.parametrize(
    "mode",
    [
        ModuleMode.DEFAULT,
        ModuleMode.COVARIANCE,
        ModuleMode.LAMBDA,
        ModuleMode.PRECONDITION_GRADIENT,
        ModuleMode.GRADIENT_AGGREGATION,
    ],
)
@pytest.mark.parametrize("train_size", [32])
@pytest.mark.parametrize("seed", [1])
def test_tracked_modules_backward_equivalence(
    test_name: str,
    mode: ModuleMode,
    train_size: int,
    seed: int,
) -> None:
    # The backward pass should produce the same results with and without wrapped modules.
    model, train_dataset, _, data_collator, task = prepare_test(
        test_name=test_name,
        train_size=train_size,
        seed=seed,
    )
    batch_size = 4
    train_loader = DataLoader(
        train_dataset,
        collate_fn=data_collator,
        batch_size=batch_size,
        drop_last=False,
        shuffle=False,
    )
    factor_args = FactorArguments(
        strategy="identity",
    )
    wrapped_model = wrap_tracked_modules(copy.deepcopy(model), factor_args=factor_args)

    original_grads = {}
    for batch in train_loader:
        model.zero_grad(set_to_none=True)
        original_loss = task.compute_train_loss(batch, model, sample=False)
        original_loss.backward()
        for name, param in model.named_parameters():
            original_grads[name] = param.grad.detach()

    set_mode(model=wrapped_model, mode=mode)
    wrapped_grads = {}
    for batch in train_loader:
        wrapped_model.zero_grad(set_to_none=True)
        wrapped_loss = task.compute_train_loss(batch, wrapped_model, sample=False)
        wrapped_loss.backward()
        for name, param in wrapped_model.named_parameters():
            if param.grad is not None:
                wrapped_grads[name] = param.grad.detach()

    for name, grad in wrapped_grads.items():
        original_name = name.replace(".original_module", "")
        if original_name in original_grads:
            assert torch.allclose(grad, original_grads[original_name])
