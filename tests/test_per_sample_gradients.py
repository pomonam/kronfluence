import copy
from typing import Any, Dict, List

import pytest
import torch
from accelerate.utils import find_batch_size
from torch import nn
from torch.utils.data import DataLoader

from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.arguments import FactorArguments
from kronfluence.module.constants import (
    LAMBDA_MATRIX_NAME,
    PRECONDITIONED_GRADIENT_NAME,
)
from kronfluence.module.tracked_module import ModuleMode, TrackedModule
from kronfluence.module.utils import set_mode, update_factor_args
from kronfluence.task import Task
from kronfluence.utils.dataset import DataLoaderKwargs
from tests.utils import (
    ATOL,
    RTOL,
    check_tensor_dict_equivalence,
    prepare_test,
    reshape_parameter_gradient_to_module_matrix,
)


def _extract_single_example(batch: Any, index: int) -> Any:
    if isinstance(batch, list):
        return [
            (
                element[index].unsqueeze(0)
                if isinstance(element[index], torch.Tensor)
                else element[index]
            )
            for element in batch
        ]
    if isinstance(batch, dict):
        return {
            key: (
                value[index].unsqueeze(0)
                if isinstance(value[index], torch.Tensor)
                else value[index]
            )
            for key, value in batch.items()
        }
    error_msg = (
        f"Unsupported batch type: {type(batch)}. Only list or dict are supported."
    )
    raise NotImplementedError(error_msg)


def for_loop_per_sample_gradient(
    batches: List[Any], model: nn.Module, task: Task, use_measurement: bool
) -> List[Dict[str, torch.Tensor]]:
    total_per_sample_gradients = []
    for batch in batches:
        parameter_gradient_dict = {}
        single_batch_list = [
            _extract_single_example(batch=batch, index=i)
            for i in range(find_batch_size(batch))
        ]
        for single_batch in single_batch_list:
            model.zero_grad(set_to_none=True)
            if use_measurement:
                loss = task.compute_measurement(
                    batch=single_batch,
                    model=model,
                )
            else:
                loss = task.compute_train_loss(
                    batch=single_batch,
                    model=model,
                    sample=False,
                )
            loss.backward()

            for param_name, param in model.named_parameters():
                if param.grad is not None:
                    if param_name not in parameter_gradient_dict:
                        parameter_gradient_dict[param_name] = []
                    parameter_gradient_dict[param_name].append(param.grad)

        for name in parameter_gradient_dict:
            parameter_gradient_dict[name] = torch.stack(parameter_gradient_dict[name])

        module_gradient_dict = {}
        for module_name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                module_gradient_dict[module_name] = (
                    reshape_parameter_gradient_to_module_matrix(
                        module=module,
                        module_name=module_name,
                        gradient_dict=parameter_gradient_dict,
                        remove_gradient=True,
                    )
                )
        del parameter_gradient_dict
        total_per_sample_gradients.append(module_gradient_dict)
    return total_per_sample_gradients


@pytest.mark.parametrize(
    "test_name",
    [
        "mlp",
        "repeated_mlp",
        "conv",
        "conv_bn",
        "bert",
    ],
)
@pytest.mark.parametrize("use_measurement", [True, False])
@pytest.mark.parametrize("train_size", [32])
@pytest.mark.parametrize("seed", [0])
def test_for_loop_per_sample_gradient_equivalence(
    test_name: str,
    use_measurement: bool,
    train_size: int,
    seed: int,
) -> None:
    model, train_dataset, _, data_collator, task = prepare_test(
        test_name=test_name,
        train_size=train_size,
        seed=seed,
    )
    original_model = copy.deepcopy(model)

    batch_size = 4
    num_batches = train_size // batch_size
    train_loader = DataLoader(
        train_dataset,
        collate_fn=data_collator,
        batch_size=batch_size,
        drop_last=False,
        shuffle=False,
    )

    batch_lst = []
    train_iter = iter(train_loader)
    for _ in range(num_batches):
        batch_lst.append(next(train_iter))

    model = prepare_model(model=model, task=task)
    analyzer = Analyzer(
        analysis_name=f"pytest_{test_name}",
        model=model,
        task=task,
        disable_model_save=True,
        cpu=True,
    )
    kwargs = DataLoaderKwargs(collate_fn=data_collator)
    factor_args = FactorArguments(
        strategy="identity",
    )
    update_factor_args(model=model, factor_args=factor_args)

    analyzer.fit_all_factors(
        factors_name=f"pytest_{test_name}_gradient",
        dataset=train_dataset,
        factor_args=factor_args,
        per_device_batch_size=1,
        overwrite_output_dir=True,
        dataloader_kwargs=kwargs,
    )

    per_sample_gradients = []
    set_mode(model, ModuleMode.PRECONDITION_GRADIENT)
    for i in range(num_batches):
        model.zero_grad(set_to_none=True)
        if use_measurement:
            loss = task.compute_measurement(
                batch=batch_lst[i],
                model=model,
            )
        else:
            loss = task.compute_train_loss(
                batch=batch_lst[i], model=model, sample=False
            )
        loss.backward()

        module_gradients = {}
        for module in model.modules():
            if isinstance(module, TrackedModule):
                module_gradients[module.name] = module.get_factor(
                    factor_name=PRECONDITIONED_GRADIENT_NAME
                )

        per_sample_gradients.append(module_gradients)

    for_loop_per_sample_gradients = for_loop_per_sample_gradient(
        batches=batch_lst,
        model=original_model,
        task=task,
        use_measurement=use_measurement,
    )
    for i in range(num_batches):
        assert check_tensor_dict_equivalence(
            per_sample_gradients[i],
            for_loop_per_sample_gradients[i],
            atol=ATOL,
            rtol=RTOL,
        )


@pytest.mark.parametrize(
    "test_name",
    [
        "mlp",
        "repeated_mlp",
        "conv",
    ],
)
@pytest.mark.parametrize("train_size", [32])
@pytest.mark.parametrize("seed", [0])
def test_lambda_equivalence(
    test_name: str,
    train_size: int,
    seed: int,
) -> None:
    model, train_dataset, _, data_collator, task = prepare_test(
        test_name=test_name,
        train_size=train_size,
        seed=seed,
    )
    original_model = copy.deepcopy(model)

    batch_size = 4
    num_batches = train_size // batch_size
    train_loader = DataLoader(
        train_dataset,
        collate_fn=data_collator,
        batch_size=batch_size,
        drop_last=False,
        shuffle=False,
    )

    batch_lst = []
    train_iter = iter(train_loader)
    for _ in range(num_batches):
        batch_lst.append(next(train_iter))

    model = prepare_model(model=model, task=task)
    analyzer = Analyzer(
        analysis_name=f"pytest_{test_name}",
        model=model,
        task=task,
        disable_model_save=True,
        cpu=True,
    )
    kwargs = DataLoaderKwargs(collate_fn=data_collator)
    factor_args = FactorArguments(
        strategy="diag",
        use_empirical_fisher=True,
    )
    analyzer.fit_lambda_matrices(
        factors_name=f"pytest_{test_name}_lambda_diag",
        dataset=train_dataset,
        factor_args=factor_args,
        per_device_batch_size=4,
        overwrite_output_dir=True,
        dataloader_kwargs=kwargs,
    )
    lambda_factors = analyzer.load_lambda_matrices(
        factors_name=f"pytest_{test_name}_lambda_diag"
    )
    lambda_matrices = lambda_factors[LAMBDA_MATRIX_NAME]

    for_loop_per_sample_gradients = for_loop_per_sample_gradient(
        batches=batch_lst,
        model=original_model,
        task=task,
        use_measurement=False,
    )
    aggregated_matrices = {}
    total_added = {}
    for gradient_batch in for_loop_per_sample_gradients:
        for module_name in gradient_batch:
            if module_name not in aggregated_matrices:
                aggregated_matrices[module_name] = (
                    gradient_batch[module_name] ** 2.0
                ).sum(dim=0)
                total_added[module_name] = gradient_batch[module_name].shape[0]
            else:
                aggregated_matrices[module_name] += (
                    gradient_batch[module_name] ** 2.0
                ).sum(dim=0)
                total_added[module_name] += gradient_batch[module_name].shape[0]
    assert check_tensor_dict_equivalence(
        lambda_matrices,
        aggregated_matrices,
        atol=ATOL,
        rtol=RTOL,
    )
