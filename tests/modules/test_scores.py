# pylint: skip-file
import time

import opt_einsum
import torch
from accelerate.utils import set_seed
from opt_einsum import DynamicProgramming


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


def test_pairwise_score_computation(
    seed: int = 0,
) -> None:
    input_dim = 4096
    output_dim = 1024
    token_dim = 512
    batch_dim = 32
    query_batch_dim = 16
    rank = 16

    set_seed(seed)

    lr_gradient1 = torch.rand(size=(query_batch_dim, output_dim, rank), dtype=torch.float64)
    lr_gradient2 = torch.rand(size=(query_batch_dim, rank, input_dim), dtype=torch.float64)
    lr_gradient = torch.bmm(lr_gradient1, lr_gradient2)

    U, S, V = torch.linalg.svd(
        lr_gradient.contiguous(),
        full_matrices=False,
    )
    U_k = U[:, :, :rank]
    S_k = S[:, :rank]
    V_k = V[:, :rank, :].clone()
    left_mat, right_mat = torch.matmul(U_k, torch.diag_embed(S_k)).contiguous(), V_k.contiguous()

    output_gradient = torch.rand(size=(batch_dim, token_dim, output_dim), dtype=torch.float64)
    input_activation = torch.rand(size=(batch_dim, token_dim, input_dim), dtype=torch.float64)

    start_time = time.time()
    train_gradient = opt_einsum.contract("b...i,b...o->bio", output_gradient, input_activation)
    gt = opt_einsum.contract("qio,bio->qb", lr_gradient, train_gradient)
    print(f"Took {time.time() - start_time} seconds.")

    start_time = time.time()
    train_gradient = opt_einsum.contract("b...i,b...o->bio", output_gradient, input_activation)
    gt_wo_einsum = lr_gradient.reshape(query_batch_dim, -1) @ train_gradient.reshape(batch_dim, -1).T
    print(f"Took {time.time() - start_time} seconds.")

    assert torch.allclose(gt, gt_wo_einsum)

    start_time = time.time()
    direct1 = opt_einsum.contract("qik,b...i,b...o,qko->qb", left_mat, output_gradient, input_activation, right_mat)
    print(f"Took {time.time() - start_time} seconds.")

    start_time = time.time()
    direct2 = opt_einsum.contract("qio,b...i,b...o->qb", lr_gradient, output_gradient, input_activation)
    print(f"Took {time.time() - start_time} seconds.")

    assert torch.allclose(gt, direct1)
    assert torch.allclose(gt, direct2)

    path1 = opt_einsum.contract_path(
        "qik,b...i,b...o,qko->qb", left_mat, output_gradient, input_activation, right_mat, optimize="optimal"
    )
    path2 = opt_einsum.contract_path(
        "qio,b...i,b...o->qb", lr_gradient, output_gradient, input_activation, optimize="optimal"
    )
    print(path1)
    print(path2)

    print("=" * 80)

    path1 = opt_einsum.contract_path(
        "qik,b...i,b...o,qko->qb", left_mat, output_gradient, input_activation, right_mat, optimize="greedy"
    )
    path2 = opt_einsum.contract_path(
        "qio,b...i,b...o->qb", lr_gradient, output_gradient, input_activation, optimize="greedy"
    )
    print(path1)
    print(path2)

    print("=" * 80)

    path1 = opt_einsum.contract_path(
        "qik,b...i,b...o,qko->qb", left_mat, output_gradient, input_activation, right_mat, optimize="dp"
    )
    path2 = opt_einsum.contract_path(
        "qio,b...i,b...o->qb", lr_gradient, output_gradient, input_activation, optimize="dp"
    )
    print(path1)
    print(path2)

    path1 = opt_einsum.contract_path(
        "qik,b...i,b...o,qko->qb",
        left_mat,
        output_gradient,
        input_activation,
        right_mat,
        optimize=DynamicProgramming(search_outer=True, minimize="size"),
    )
    path2 = opt_einsum.contract_path(
        "qio,b...i,b...o->qb",
        lr_gradient,
        output_gradient,
        input_activation,
        optimize=DynamicProgramming(search_outer=True, minimize="size"),
    )
    print(path1)
    print(path2)

    # path1 = opt_einsum.contract_path("qik,b...i,b...o,qko->qb", left_mat, output_gradient, input_activation, right_mat)
    # path2 = opt_einsum.contract_path("qio,b...i,b...o->qb", lr_gradient, output_gradient, input_activation)
    # print(path1)
    # print(path2)
