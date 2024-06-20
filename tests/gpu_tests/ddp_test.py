# pylint: skip-file

import logging
import os
import unittest

import torch
import torch.distributed as dist
from torch.utils import data

from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.arguments import FactorArguments, ScoreArguments
from kronfluence.utils.constants import (
    ALL_MODULE_NAME,
    COVARIANCE_FACTOR_NAMES,
    LAMBDA_FACTOR_NAMES,
)
from kronfluence.utils.model import apply_ddp
from tests.gpu_tests.pipeline import GpuTestTask, construct_test_mlp, get_mnist_dataset
from tests.gpu_tests.prepare_tests import QUERY_INDICES, TRAIN_INDICES
from tests.utils import check_tensor_dict_equivalence

LOCAL_RANK = int(os.environ["LOCAL_RANK"])
WORLD_RANK = int(os.environ["RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
logging.basicConfig(level=logging.DEBUG)
OLD_FACTOR_NAME = "single_gpu"
NEW_FACTOR_NAME = "ddp"
OLD_SCORE_NAME = "single_gpu"
NEW_SCORE_NAME = "ddp"


class DDPTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = construct_test_mlp()
        cls.model.load_state_dict(torch.load("model.pth"))
        cls.model = cls.model.double()

        cls.train_dataset = get_mnist_dataset(split="train", data_path="data")
        cls.train_dataset = data.Subset(cls.train_dataset, indices=list(range(TRAIN_INDICES)))
        cls.eval_dataset = get_mnist_dataset(split="valid", data_path="data")
        cls.eval_dataset = data.Subset(cls.eval_dataset, indices=list(range(QUERY_INDICES)))

        cls.task = GpuTestTask()
        cls.model = prepare_model(cls.model, cls.task)

        cls.model = apply_ddp(
            model=cls.model,
            local_rank=LOCAL_RANK,
            rank=WORLD_RANK,
            world_size=WORLD_SIZE,
        )

        cls.analyzer = Analyzer(
            analysis_name="gpu_test",
            model=cls.model,
            task=cls.task,
        )

    def test_covariance_matrices(self) -> None:
        covariance_factors = self.analyzer.load_covariance_matrices(factors_name=OLD_FACTOR_NAME)
        factor_args = FactorArguments(
            use_empirical_fisher=True,
            activation_covariance_dtype=torch.float64,
            gradient_covariance_dtype=torch.float64,
            lambda_dtype=torch.float64,
        )
        self.analyzer.fit_covariance_matrices(
            factors_name=NEW_FACTOR_NAME,
            dataset=self.train_dataset,
            factor_args=factor_args,
            per_device_batch_size=512,
            overwrite_output_dir=True,
        )
        new_covariance_factors = self.analyzer.load_covariance_matrices(factors_name=NEW_FACTOR_NAME)

        for name in COVARIANCE_FACTOR_NAMES:
            if LOCAL_RANK == 0:
                for module_name in covariance_factors[name]:
                    print(f"Name: {name, module_name}")
                    print(f"Previous factor: {covariance_factors[name][module_name]}")
                    print(f"New factor: {new_covariance_factors[name][module_name]}")
            if LOCAL_RANK == 0:
                assert check_tensor_dict_equivalence(
                    covariance_factors[name],
                    new_covariance_factors[name],
                    atol=1e-5,
                    rtol=1e-3,
                )

    def test_lambda_matrices(self):
        lambda_factors = self.analyzer.load_lambda_matrices(factors_name=OLD_FACTOR_NAME)
        factor_args = FactorArguments(
            use_empirical_fisher=True,
            activation_covariance_dtype=torch.float64,
            gradient_covariance_dtype=torch.float64,
            lambda_dtype=torch.float64,
        )
        self.analyzer.fit_lambda_matrices(
            factors_name=NEW_FACTOR_NAME,
            dataset=self.train_dataset,
            factor_args=factor_args,
            per_device_batch_size=512,
            overwrite_output_dir=True,
            load_from_factors_name=OLD_FACTOR_NAME,
        )
        new_lambda_factors = self.analyzer.load_lambda_matrices(factors_name=NEW_FACTOR_NAME)

        for name in LAMBDA_FACTOR_NAMES:
            if LOCAL_RANK == 0:
                for module_name in lambda_factors[name]:
                    print(f"Name: {name, module_name}")
                    print(f"Previous factor: {lambda_factors[name][module_name]}")
                    print(f"New factor: {new_lambda_factors[name][module_name]}")
            if LOCAL_RANK == 0:
                assert check_tensor_dict_equivalence(
                    lambda_factors[name],
                    new_lambda_factors[name],
                    atol=1e-3,
                    rtol=1e-1,
                )

    def test_pairwise_scores(self) -> None:
        pairwise_scores = self.analyzer.load_pairwise_scores(scores_name=OLD_SCORE_NAME)

        score_args = ScoreArguments(
            score_dtype=torch.float64,
            per_sample_gradient_dtype=torch.float64,
            precondition_dtype=torch.float64,
        )
        self.analyzer.compute_pairwise_scores(
            scores_name=NEW_SCORE_NAME,
            factors_name=OLD_FACTOR_NAME,
            query_dataset=self.eval_dataset,
            train_dataset=self.train_dataset,
            train_indices=list(range(TRAIN_INDICES)),
            query_indices=list(range(QUERY_INDICES)),
            per_device_query_batch_size=12,
            per_device_train_batch_size=512,
            score_args=score_args,
            overwrite_output_dir=True,
        )
        new_pairwise_scores = self.analyzer.load_pairwise_scores(scores_name=NEW_SCORE_NAME)

        if LOCAL_RANK == 0:
            print(f"Previous score: {pairwise_scores[ALL_MODULE_NAME][0]}")
            print(f"Previous shape: {pairwise_scores[ALL_MODULE_NAME].shape}")
            print(f"New score: {new_pairwise_scores[ALL_MODULE_NAME][0]}")
            print(f"New shape: {new_pairwise_scores[ALL_MODULE_NAME].shape}")
            print(f"Previous score: {pairwise_scores[ALL_MODULE_NAME][50]}")
            print(f"Previous shape: {pairwise_scores[ALL_MODULE_NAME].shape}")
            print(f"New score: {new_pairwise_scores[ALL_MODULE_NAME][50]}")
            print(f"New shape: {new_pairwise_scores[ALL_MODULE_NAME].shape}")
            assert check_tensor_dict_equivalence(
                pairwise_scores,
                new_pairwise_scores,
                atol=1e-5,
                rtol=1e-3,
            )

    def test_self_scores(self) -> None:
        self_scores = self.analyzer.load_self_scores(scores_name=OLD_SCORE_NAME)

        score_args = ScoreArguments(
            score_dtype=torch.float64,
            per_sample_gradient_dtype=torch.float64,
            precondition_dtype=torch.float64,
        )
        self.analyzer.compute_self_scores(
            scores_name=NEW_SCORE_NAME,
            factors_name=OLD_FACTOR_NAME,
            train_dataset=self.train_dataset,
            train_indices=list(range(TRAIN_INDICES)),
            per_device_train_batch_size=512,
            score_args=score_args,
            overwrite_output_dir=True,
        )
        new_self_scores = self.analyzer.load_self_scores(scores_name=NEW_SCORE_NAME)

        if LOCAL_RANK == 0:
            print(f"Previous score: {self_scores[ALL_MODULE_NAME]}")
            print(f"Previous shape: {self_scores[ALL_MODULE_NAME].shape}")
            print(f"New score: {new_self_scores[ALL_MODULE_NAME]}")
            print(f"New shape: {new_self_scores[ALL_MODULE_NAME].shape}")
            assert check_tensor_dict_equivalence(
                self_scores,
                new_self_scores,
                atol=1e-5,
                rtol=1e-3,
            )

    def test_lr_pairwise_scores(self) -> None:
        pairwise_scores = self.analyzer.load_pairwise_scores(scores_name="single_gpu_qb")

        score_args = ScoreArguments(
            query_gradient_rank=32,
            score_dtype=torch.float64,
            per_sample_gradient_dtype=torch.float64,
            precondition_dtype=torch.float64,
            query_gradient_svd_dtype=torch.float64,
        )
        self.analyzer.compute_pairwise_scores(
            scores_name="ddp_qb",
            factors_name=OLD_FACTOR_NAME,
            query_dataset=self.eval_dataset,
            train_dataset=self.train_dataset,
            train_indices=list(range(TRAIN_INDICES)),
            query_indices=list(range(QUERY_INDICES)),
            per_device_query_batch_size=12,
            per_device_train_batch_size=512,
            score_args=score_args,
            overwrite_output_dir=True,
        )
        new_pairwise_scores = self.analyzer.load_pairwise_scores(scores_name="ddp_qb")

        if LOCAL_RANK == 0:
            print(f"Previous score: {pairwise_scores[ALL_MODULE_NAME][0]}")
            print(f"Previous shape: {pairwise_scores[ALL_MODULE_NAME].shape}")
            print(f"New score: {new_pairwise_scores[ALL_MODULE_NAME][0]}")
            print(f"New shape: {new_pairwise_scores[ALL_MODULE_NAME].shape}")
            print(f"Previous score: {pairwise_scores[ALL_MODULE_NAME][50]}")
            print(f"Previous shape: {pairwise_scores[ALL_MODULE_NAME].shape}")
            print(f"New score: {new_pairwise_scores[ALL_MODULE_NAME][50]}")
            print(f"New shape: {new_pairwise_scores[ALL_MODULE_NAME].shape}")
            assert check_tensor_dict_equivalence(
                pairwise_scores,
                new_pairwise_scores,
                atol=1e-3,
                rtol=1e-1,
            )

    def test_per_module_pairwise_scores(self) -> None:
        score_args = ScoreArguments(
            score_dtype=torch.float64,
            per_sample_gradient_dtype=torch.float64,
            precondition_dtype=torch.float64,
            query_gradient_svd_dtype=torch.float64,
            per_module_score=True,
        )
        self.analyzer.compute_pairwise_scores(
            scores_name="ddp_per_module",
            factors_name=OLD_FACTOR_NAME,
            query_dataset=self.eval_dataset,
            train_dataset=self.train_dataset,
            train_indices=list(range(TRAIN_INDICES)),
            query_indices=list(range(QUERY_INDICES)),
            per_device_query_batch_size=11,
            per_device_train_batch_size=512,
            score_args=score_args,
            overwrite_output_dir=True,
        )

    def test_lr_aggregate_pairwise_scores(self) -> None:
        pairwise_scores = self.analyzer.load_pairwise_scores(scores_name="single_gpu_qb")

        score_args = ScoreArguments(
            query_gradient_rank=32,
            num_query_gradient_aggregations=3,
            score_dtype=torch.float64,
            per_sample_gradient_dtype=torch.float64,
            precondition_dtype=torch.float64,
            query_gradient_svd_dtype=torch.float64,
        )
        self.analyzer.compute_pairwise_scores(
            scores_name="ddp_qb_agg",
            factors_name=OLD_FACTOR_NAME,
            query_dataset=self.eval_dataset,
            train_dataset=self.train_dataset,
            train_indices=list(range(TRAIN_INDICES)),
            query_indices=list(range(QUERY_INDICES)),
            per_device_query_batch_size=2,
            per_device_train_batch_size=512,
            score_args=score_args,
            overwrite_output_dir=True,
        )
        new_pairwise_scores = self.analyzer.load_pairwise_scores(scores_name="ddp_qb_agg")

        if LOCAL_RANK == 0:
            print(f"Previous score: {pairwise_scores[ALL_MODULE_NAME][0]}")
            print(f"Previous shape: {pairwise_scores[ALL_MODULE_NAME].shape}")
            print(f"New score: {new_pairwise_scores[ALL_MODULE_NAME][0]}")
            print(f"New shape: {new_pairwise_scores[ALL_MODULE_NAME].shape}")
            print(f"Previous score: {pairwise_scores[ALL_MODULE_NAME][44]}")
            print(f"New score: {new_pairwise_scores[ALL_MODULE_NAME][44]}")
            assert check_tensor_dict_equivalence(
                pairwise_scores,
                new_pairwise_scores,
                atol=1e-1,
                rtol=1e-1,
            )

    @classmethod
    def tearDownClass(cls) -> None:
        dist.destroy_process_group()


if __name__ == "__main__":
    unittest.main()
