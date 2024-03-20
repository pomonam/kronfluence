# pylint: skip-file

import logging
import os
import unittest

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.arguments import FactorArguments, ScoreArguments
from kronfluence.task import Task
from tests.gpu_tests.ddp_test import OLD_FACTOR_NAME
from tests.gpu_tests.pipeline import BATCH_TYPE, construct_test_mlp, get_mnist_dataset

LOCAL_RANK = int(os.environ["LOCAL_RANK"])
WORLD_RANK = int(os.environ["RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
logging.basicConfig(level=logging.DEBUG)
NEW_FACTOR_NAME = "ddp_variation"
NEW_SCORE_NAME = "ddp_variation"


class GpuVariationTask(Task):
    def compute_train_loss(
        self,
        batch: BATCH_TYPE,
        model: nn.Module,
        sample: bool = False,
    ) -> torch.Tensor:
        inputs, labels = batch
        logits = model(inputs)
        if not sample:
            return F.cross_entropy(logits, labels, reduction="sum")
        with torch.no_grad():
            probs = torch.nn.functional.softmax(logits, dim=-1)
            sampled_labels = torch.multinomial(
                probs,
                num_samples=1,
            ).flatten()
        return F.cross_entropy(logits, sampled_labels.detach(), reduction="sum")

    def compute_measurement(
        self,
        batch: BATCH_TYPE,
        model: nn.Module,
    ) -> torch.Tensor:
        inputs, labels = batch
        logits = model(inputs)

        bindex = torch.arange(logits.shape[0]).to(device=logits.device, non_blocking=False)
        logits_correct = logits[bindex, labels]

        cloned_logits = logits.clone()
        cloned_logits[bindex, labels] = torch.tensor(-torch.inf, device=logits.device, dtype=logits.dtype)

        margins = logits_correct - cloned_logits.logsumexp(dim=-1)
        return -margins.sum()


class DDPVariationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = construct_test_mlp()
        cls.model.load_state_dict(torch.load("model.pth"))

        cls.train_dataset = get_mnist_dataset(split="train", data_path="data")
        cls.eval_dataset = get_mnist_dataset(split="valid", data_path="data")

        cls.task = GpuVariationTask()
        cls.model = prepare_model(cls.model, cls.task)

        dist.init_process_group("nccl", rank=WORLD_RANK, world_size=WORLD_SIZE)
        device = torch.device("cuda:{}".format(LOCAL_RANK))
        torch.cuda.set_device(LOCAL_RANK)

        cls.model = cls.model.to(device=device)
        cls.model = DistributedDataParallel(cls.model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)
        cls.analyzer = Analyzer(
            analysis_name="gpu_test",
            model=cls.model,
            task=cls.task,
        )

    def test_covariance_matrices(self) -> None:
        data_partition_lst = [1, 3]
        module_partition_lst = [1, 3]
        cached_activation_cpu_offload = [False, True]

        for dp in data_partition_lst:
            for mp in module_partition_lst:
                for ca in cached_activation_cpu_offload:
                    factor_args = FactorArguments(
                        covariance_data_partition_size=dp,
                        covariance_module_partition_size=mp,
                        cached_activation_cpu_offload=ca,
                    )
                    self.analyzer.fit_covariance_matrices(
                        factors_name=NEW_FACTOR_NAME,
                        dataset=self.train_dataset,
                        factor_args=factor_args,
                        per_device_batch_size=512,
                        overwrite_output_dir=True,
                    )

    def test_lambda_matrices(self):
        data_partition_lst = [1, 3]
        module_partition_lst = [1, 3]
        cached_activation_cpu_offload = [False, True]

        for dp in data_partition_lst:
            for mp in module_partition_lst:
                for ca in cached_activation_cpu_offload:
                    factor_args = FactorArguments(
                        lambda_data_partition_size=dp,
                        lambda_module_partition_size=mp,
                        cached_activation_cpu_offload=ca,
                    )
                    self.analyzer.fit_lambda_matrices(
                        factors_name=NEW_FACTOR_NAME,
                        dataset=self.train_dataset,
                        factor_args=factor_args,
                        per_device_batch_size=512,
                        overwrite_output_dir=True,
                        load_from_factors_name=OLD_FACTOR_NAME,
                    )

    def test_pairwise_scores(self) -> None:
        score_args = ScoreArguments(
            data_partition_size=3,
            module_partition_size=3,
        )
        self.analyzer.compute_pairwise_scores(
            scores_name=NEW_SCORE_NAME,
            factors_name=OLD_FACTOR_NAME,
            query_dataset=self.eval_dataset,
            train_dataset=self.train_dataset,
            per_device_query_batch_size=256,
            per_device_train_batch_size=512,
            score_args=score_args,
            overwrite_output_dir=True,
        )

    def test_self_scores(self) -> None:
        score_args = ScoreArguments(
            data_partition_size=3,
            module_partition_size=3,
        )
        self.analyzer.compute_self_scores(
            scores_name=NEW_SCORE_NAME,
            factors_name=OLD_FACTOR_NAME,
            train_dataset=self.train_dataset,
            per_device_train_batch_size=512,
            score_args=score_args,
            overwrite_output_dir=True,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        dist.destroy_process_group()


if __name__ == "__main__":
    unittest.main()
