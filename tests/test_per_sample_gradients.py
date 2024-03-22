# pylint: skip-file

import copy
import time
from typing import Any, Dict, List

import opt_einsum
import pytest
import torch
from accelerate.utils import find_batch_size, set_seed
from torch import nn
from torch.utils.data import DataLoader

from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.arguments import FactorArguments
from kronfluence.module.tracked_module import ModuleMode, TrackedModule
from kronfluence.module.utils import set_mode, update_factor_args
from kronfluence.task import Task
from kronfluence.utils.constants import LAMBDA_MATRIX_NAME, PRECONDITIONED_GRADIENT_NAME
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
            (element[index].unsqueeze(0) if isinstance(element[index], torch.Tensor) else element[index])
            for element in batch
        ]
    if isinstance(batch, dict):
        return {
            key: (value[index].unsqueeze(0) if isinstance(value[index], torch.Tensor) else value[index])
            for key, value in batch.items()
        }
    error_msg = f"Unsupported batch type: {type(batch)}. Only list or dict are supported."
    raise NotImplementedError(error_msg)


def for_loop_per_sample_gradient(
    batches: List[Any], model: nn.Module, task: Task, use_measurement: bool
) -> List[Dict[str, torch.Tensor]]:
    total_per_sample_gradients = []
    for batch in batches:
        parameter_gradient_dict = {}
        single_batch_list = [_extract_single_example(batch=batch, index=i) for i in range(find_batch_size(batch))]
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
                module_gradient_dict[module_name] = reshape_parameter_gradient_to_module_matrix(
                    module=module,
                    module_name=module_name,
                    gradient_dict=parameter_gradient_dict,
                    remove_gradient=True,
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
        "gpt",
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
            loss = task.compute_train_loss(batch=batch_lst[i], model=model, sample=False)
        loss.backward()

        module_gradients = {}
        for module in model.modules():
            if isinstance(module, TrackedModule):
                module_gradients[module.name] = module.get_factor(factor_name=PRECONDITIONED_GRADIENT_NAME)

        per_sample_gradients.append(module_gradients)

    for_loop_per_sample_gradients = for_loop_per_sample_gradient(
        batches=batch_lst,
        model=original_model,
        task=task,
        use_measurement=use_measurement,
    )

    for i in range(num_batches):
        if "lm_head" in for_loop_per_sample_gradients[i]:
            del for_loop_per_sample_gradients[i]["lm_head"]

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
        "gpt",
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
        strategy="diagonal",
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
    lambda_factors = analyzer.load_lambda_matrices(factors_name=f"pytest_{test_name}_lambda_diag")
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
                aggregated_matrices[module_name] = (gradient_batch[module_name] ** 2.0).sum(dim=0)
                total_added[module_name] = gradient_batch[module_name].shape[0]
            else:
                aggregated_matrices[module_name] += (gradient_batch[module_name] ** 2.0).sum(dim=0)
                total_added[module_name] += gradient_batch[module_name].shape[0]
    assert check_tensor_dict_equivalence(
        lambda_matrices,
        aggregated_matrices,
        atol=ATOL,
        rtol=RTOL,
    )


def test_precondition_gradient(
    seed: int = 0,
) -> None:
    input_dim = 128
    output_dim = 256
    batch_dim = 8
    lambda_scale = 1000
    damping = 1e-08

    set_seed(seed)
    A = torch.rand(size=(input_dim, input_dim), dtype=torch.float64)
    B = torch.rand(size=(output_dim, output_dim), dtype=torch.float64)
    Lambda = torch.rand(size=(output_dim, input_dim), dtype=torch.float64)

    gradient = torch.rand(size=(batch_dim, output_dim, input_dim), dtype=torch.float64)

    start_time = time.time()
    rotated_gradient = torch.einsum(
        "ij,bjl,lk->bik",
        (
            B.t(),
            gradient,
            A,
        ),
    )
    rotated_gradient.div_(Lambda + damping)
    results = lambda_scale * torch.einsum(
        "ij,bjl,lk->bik",
        (B, rotated_gradient, A.t()),
    )
    print(f"Took {time.time() - start_time} seconds.")

    start_time = time.time()
    grads_rot = torch.matmul(
        B.t(),
        torch.matmul(
            gradient,
            A,
        ),
    )
    scaled_lambda = Lambda / lambda_scale
    grads_rot.div_(scaled_lambda)
    raw_results = torch.matmul(
        B,
        torch.matmul(
            grads_rot,
            A.t(),
        ),
    )
    print(f"Took {time.time() - start_time} seconds.")

    assert torch.allclose(raw_results, results, atol=1e-5, rtol=1e-3)


def test_query_gradient_svd(
    seed: int = 0,
) -> None:
    input_dim = 2048
    output_dim = 1024
    batch_dim = 16
    set_seed(seed)

    gradient = torch.rand(size=(batch_dim, output_dim, input_dim), dtype=torch.float64)

    U, S, V = torch.linalg.svd(
        gradient.contiguous(),
        full_matrices=False,
    )
    assert torch.allclose(gradient, U @ torch.diag_embed(S) @ V, atol=1e-5, rtol=1e-3)

    rank = 32
    U_k = U[:, :, :rank]
    S_k = S[:, :rank]
    V_k = V[:, :rank, :].clone()
    left, right = torch.matmul(U_k, torch.diag_embed(S_k)).contiguous(), V_k.contiguous()
    assert torch.bmm(left, right).shape == gradient.shape

    rank = input_dim
    U, S, V = torch.linalg.svd(
        gradient.contiguous(),
        full_matrices=False,
    )
    U_k = U[:, :, :rank]
    S_k = S[:, :rank]
    V_k = V[:, :rank, :].clone()
    left, right = torch.matmul(U_k, torch.diag_embed(S_k)).contiguous(), V_k.contiguous()
    assert torch.allclose(torch.bmm(left, right), gradient, atol=1e-5, rtol=1e-3)

    rank = 32
    lr_gradient1 = torch.rand(size=(batch_dim, output_dim, rank), dtype=torch.float64)
    lr_gradient2 = torch.rand(size=(batch_dim, rank, input_dim), dtype=torch.float64)
    gradient = torch.bmm(lr_gradient1, lr_gradient2)
    U, S, V = torch.linalg.svd(
        gradient.contiguous(),
        full_matrices=False,
    )
    U_k = U[:, :, :rank]
    S_k = S[:, :rank]
    V_k = V[:, :rank, :].clone()
    left_mat, right_mat = torch.matmul(U_k, torch.diag_embed(S_k)).contiguous(), V_k.contiguous()
    assert torch.allclose(torch.bmm(left_mat, right_mat), gradient, atol=1e-5, rtol=1e-3)

    query_batch_dim = 32
    new_gradient = torch.rand(size=(query_batch_dim, output_dim, input_dim), dtype=torch.float64)
    score = opt_einsum.contract("toi,qoi->tq", gradient, new_gradient)

    lr_score = opt_einsum.contract("qki,toi,qok->qt", right_mat, new_gradient, left_mat)
    assert torch.allclose(score, lr_score)

    lr_score_reconst_matmul = torch.matmul(
        torch.matmul(left_mat, right_mat).view(left_mat.size(0), -1), new_gradient.view(new_gradient.shape[0], -1).t()
    )
    assert torch.allclose(score, lr_score_reconst_matmul)

    # These should be able to avoid explicit reconstruction.
    # This should be used when input_dim > output_dim.
    intermediate = opt_einsum.contract("qki,toi->qtko", right_mat, new_gradient)
    final = opt_einsum.contract("qtko,qok->qt", intermediate, left_mat)
    assert torch.allclose(score, final)
    print("Option 1")
    print(intermediate.numel())

    # This should be used when output_dim > input_dim.
    intermediate2 = torch.einsum("toi,qok->qtik", new_gradient, left_mat)
    final2 = opt_einsum.contract("qki,qtik->qt", right_mat, intermediate2)
    assert torch.allclose(score, final2)
    print("Option 2")
    print(intermediate2.numel())

    print("Reconstruction")
    print((torch.matmul(left_mat, right_mat).view(left_mat.size(0), -1)).numel())
    path = opt_einsum.contract_path("qki,toi,qok->qt", right_mat, new_gradient, left_mat)
    print(path)


@pytest.mark.parametrize("input_dim", [256, 512])
@pytest.mark.parametrize("output_dim", [512, 1024])
@pytest.mark.parametrize("batch_dim", [8, 16])
@pytest.mark.parametrize("qbatch_dim", [8, 16])
@pytest.mark.parametrize("rank", [32])
@pytest.mark.parametrize("seed", [0])
def test_query_gradient_svd_reconst(
    input_dim: int,
    output_dim: int,
    batch_dim: int,
    qbatch_dim: int,
    rank: int,
    seed: int,
) -> None:
    set_seed(seed)

    lr_gradient1 = torch.rand(size=(batch_dim, output_dim, rank + 50), dtype=torch.float64)
    lr_gradient2 = torch.rand(size=(batch_dim, rank + 50, input_dim), dtype=torch.float64)
    gradient = torch.bmm(lr_gradient1, lr_gradient2)
    U, S, V = torch.linalg.svd(
        gradient.contiguous(),
        full_matrices=False,
    )
    U_k = U[:, :, :rank]
    S_k = S[:, :rank]
    V_k = V[:, :rank, :].clone()
    left_mat, right_mat = torch.matmul(U_k, torch.diag_embed(S_k)).contiguous(), V_k.contiguous()
    new_gradient = torch.rand(size=(qbatch_dim, output_dim, input_dim), dtype=torch.float64)

    lr_score = opt_einsum.contract("qki,toi,qok->qt", right_mat, new_gradient, left_mat)
    lr_score_reconst_matmul = torch.matmul(
        torch.matmul(left_mat, right_mat).view(left_mat.size(0), -1), new_gradient.view(new_gradient.shape[0], -1).t()
    )
    assert torch.allclose(lr_score, lr_score_reconst_matmul)

    # This should be used when input_dim > output_dim.
    intermediate = opt_einsum.contract("qki,toi->qtko", right_mat, new_gradient)
    final = opt_einsum.contract("qtko,qok->qt", intermediate, left_mat)
    assert torch.allclose(lr_score, final)
    print("Option 1")
    print(intermediate.numel())

    # This should be used when output_dim > input_dim.
    intermediate2 = torch.einsum("toi,qok->qtik", new_gradient, left_mat)
    final2 = opt_einsum.contract("qki,qtik->qt", right_mat, intermediate2)
    assert torch.allclose(lr_score, final2)
    print("Option 2")
    print(intermediate2.numel())

    print("Reconstruction")
    reconst_numel = (torch.matmul(left_mat, right_mat).view(left_mat.size(0), -1)).numel()
    print(reconst_numel)
    path = opt_einsum.contract_path("qki,toi,qok->qt", right_mat, new_gradient, left_mat)
    print(path)

    if new_gradient.size(0) * right_mat.size(0) * rank * min((right_mat.size(2), left_mat.size(1))) > right_mat.size(
        0
    ) * right_mat.size(2) * left_mat.size(1):
        assert intermediate2.numel() > reconst_numel and intermediate.numel() > reconst_numel
    elif output_dim >= input_dim:
        assert intermediate2.numel() <= reconst_numel
    else:
        assert intermediate.numel() <= reconst_numel


def test_compute_score_matmul(
    seed: int = 0,
) -> None:
    input_dim = 1024
    output_dim = 2048
    batch_dim = 16
    query_batch_dim = 64
    set_seed(seed)

    gradient = torch.rand(size=(batch_dim, output_dim, input_dim), dtype=torch.float64)
    new_gradient = torch.rand(size=(query_batch_dim, output_dim, input_dim), dtype=torch.float64)

    score = opt_einsum.contract("toi,qoi->tq", gradient, new_gradient)
    path = opt_einsum.contract_path("toi,qoi->tq", gradient, new_gradient)
    print(path)

    unsqueeze_score = opt_einsum.contract("t...,q...->tq", gradient, new_gradient)
    assert torch.allclose(score, unsqueeze_score)
    path = opt_einsum.contract_path("t...,q...->tq", gradient, new_gradient)
    print(path)
