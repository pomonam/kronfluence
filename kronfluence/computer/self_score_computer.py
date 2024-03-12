import os
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch.utils import data

from kronfluence.arguments import FactorArguments, ScoreArguments
from kronfluence.computer.computer import Computer
from kronfluence.factor.config import FactorConfig
from kronfluence.module.constants import FACTOR_TYPE, SCORE_TYPE
from kronfluence.module.tracked_module import ModuleMode
from kronfluence.module.utils import set_mode
from kronfluence.score.self import (
    compute_self_scores_with_loaders,
    load_self_scores,
    save_self_scores,
    self_scores_exist,
)
from kronfluence.utils.dataset import DataLoaderKwargs, find_executable_batch_size
from kronfluence.utils.exceptions import FactorsNotFoundError
from kronfluence.utils.logger import get_time
from kronfluence.utils.save import FACTOR_ARGUMENTS_NAME, SCORE_ARGUMENTS_NAME
from kronfluence.utils.state import release_memory


class SelfScoreComputer(Computer):
    """Handles the computation of self-influence scores for a given PyTorch model."""

    def _find_executable_self_scores_batch_size(
        self,
        loaded_factors: FACTOR_TYPE,
        train_dataset: data.Dataset,
        total_data_examples: int,
        dataloader_params: Dict[str, Any],
        score_args: ScoreArguments,
        factor_args: FactorArguments,
        tracked_modules_name: Optional[List[str]],
    ) -> int:
        """Automatically finds executable training batch size for computing self-influence scores."""
        if self.state.num_processes > 1:
            error_msg = (
                "Automatic batch size search is currently not supported for multi-GPU training. "
                "Please manually configure the batch size."
            )
            self.logger.error(error_msg)
            raise NotImplementedError(error_msg)

        self.logger.info("Automatically determining executable batch size.")
        start_batch_size = min(
            [
                score_args.initial_per_device_batch_size_attempt,
                total_data_examples,
            ]
        )

        def executable_batch_size_func(batch_size: int) -> None:
            self.logger.info(f"Attempting to set per-device batch size to {batch_size}.")
            set_mode(model=self.model, mode=ModuleMode.DEFAULT, keep_factors=False)
            release_memory()
            total_batch_size = batch_size * self.state.num_processes
            train_loader = self._get_dataloader(
                dataset=train_dataset,
                per_device_batch_size=batch_size,
                indices=list(range(total_batch_size)),
                allow_duplicates=True,
                stack=True,
                dataloader_params=dataloader_params,
            )
            compute_self_scores_with_loaders(
                model=self.model,
                state=self.state,
                task=self.task,
                loaded_factors=loaded_factors,
                train_loader=train_loader,
                score_args=score_args,
                factor_args=factor_args,
                tracked_module_names=tracked_modules_name,
            )

        per_device_batch_size = find_executable_batch_size(
            func=executable_batch_size_func,
            start_batch_size=start_batch_size,
        )
        self.logger.info(f"Executable batch size determined: {per_device_batch_size}.")
        return per_device_batch_size

    def _fit_partitioned_self_scores(
        self,
        loaded_factors: Optional[FACTOR_TYPE],
        train_dataset: data.Dataset,
        per_device_train_batch_size: int,
        dataloader_params: Dict[str, Any],
        score_args: ScoreArguments,
        factor_args: FactorArguments,
        indices: Optional[List[int]] = None,
        tracked_module_names: Optional[List[str]] = None,
    ) -> SCORE_TYPE:
        """Fits all self-influence scores for the given data and module partition."""
        release_memory()
        start_time = get_time(state=self.state)
        with self.profiler.profile("Compute Self-Influence Score"):
            train_loader = self._get_dataloader(
                dataset=train_dataset,
                per_device_batch_size=per_device_train_batch_size,
                indices=indices,
                allow_duplicates=True,
                stack=True,
                dataloader_params=dataloader_params,
            )
            scores = compute_self_scores_with_loaders(
                model=self.model,
                state=self.state,
                task=self.task,
                loaded_factors=loaded_factors,
                train_loader=train_loader,
                score_args=score_args,
                factor_args=factor_args,
                tracked_module_names=tracked_module_names,
            )
        end_time = get_time(state=self.state)
        elapsed_time = end_time - start_time
        self.logger.info(f"Computed self-influence scores in {elapsed_time:.2f} seconds.")
        return scores

    def compute_self_scores(
        self,
        scores_name: str,
        factors_name: str,
        train_dataset: data.Dataset,
        per_device_train_batch_size: Optional[int] = None,
        train_indices: Optional[Sequence[int]] = None,
        dataloader_kwargs: Optional[DataLoaderKwargs] = None,
        score_args: Optional[ScoreArguments] = None,
        target_data_partitions: Optional[Sequence[int]] = None,
        target_module_partitions: Optional[Sequence[int]] = None,
        overwrite_output_dir: bool = False,
    ) -> None:
        """Computes self-influence scores for the given score configuration. As an example,
        for T training dataset, the self-influence scores are represented as T-dimensional vector.

        Args:
            scores_name (str):
                The unique identifier for the score, used to organize and retrieve the results.
            factors_name (str):
                The name of the factor to use for influence computations.
            train_dataset (data.Dataset):
                The training dataset.
            per_device_train_batch_size (int, optional):
                The per-device batch size used to compute training gradients. If not specified, an executable
                batch size will be found.
            train_indices (Sequence[int], optional):
                The specific indices of the training dataset to compute the influence scores for. If not
                specified, all training data points will be used.
            dataloader_kwargs (DataLoaderKwargs, optional):
                Controls additional arguments for PyTorch's DataLoader.
            score_args (ScoreArguments, optional):
                Arguments related to computing the self-influence scores. If not specified, the default values
                of `ScoreArguments` will be used.
            target_data_partitions (Sequence[int], optional):
                Specific data partitions to compute influence scores. If not specified, scores for all
                data partitions will be computed.
            target_module_partitions (Sequence[int], optional):
                Specific module partitions to compute influence scores. If not specified, scores for all
                module partitions will be computed.
            overwrite_output_dir (bool, optional):
                If True, the existing factors with the same name will be overwritten.
        """
        self.logger.debug(f"Computing self-influence scores with parameters: {locals()}")

        scores_output_dir = self.scores_output_dir(scores_name=scores_name)
        os.makedirs(scores_output_dir, exist_ok=True)
        if self_scores_exist(output_dir=scores_output_dir) and not overwrite_output_dir:
            self.logger.info(f"Found existing self-influence scores at {scores_output_dir}. Skipping.")
            return

        if score_args is None:
            score_args = ScoreArguments()
            self.logger.info(f"Score arguments not provided. Using the default configuration: {score_args}.")
        else:
            self.logger.info(f"Using the provided configuration: {score_args}.")

        factor_args = self.load_factor_args(factors_name=factors_name)
        factors_output_dir = self.factors_output_dir(factors_name=factors_name)
        if factor_args is None:
            error_msg = f"Factors with name `{factors_name}` was not found at {factors_output_dir}."
            self.logger.error(error_msg)
            raise FactorsNotFoundError(error_msg)
        factor_args = FactorArguments(**factor_args)
        self.logger.info(f"Loaded FactorArguments with configuration: {factor_args}.")
        strategy = factor_args.strategy
        factor_config = FactorConfig.CONFIGS[strategy]

        if self.state.is_main_process:
            self._save_dataset_metadata(
                dataset_name="train",
                dataset=train_dataset,
                indices=train_indices,
                output_dir=scores_output_dir,
                overwrite_output_dir=overwrite_output_dir,
            )
            self._save_arguments(
                arguments_name=SCORE_ARGUMENTS_NAME,
                arguments=score_args,
                output_dir=scores_output_dir,
                overwrite_output_dir=overwrite_output_dir,
            )
            self._save_arguments(
                arguments_name=FACTOR_ARGUMENTS_NAME,
                arguments=factor_args,
                output_dir=scores_output_dir,
                overwrite_output_dir=overwrite_output_dir,
            )

        if dataloader_kwargs is None:
            dataloader_kwargs = DataLoaderKwargs()
            self.logger.info(
                f"DataLoader arguments not provided. Using the default configuration: {dataloader_kwargs}."
            )
        else:
            self.logger.info(f"Using the DataLoader parameters: {dataloader_kwargs.to_dict()}.")
        dataloader_params = dataloader_kwargs.to_dict()
        if train_indices is not None:
            train_dataset = data.Subset(dataset=train_dataset, indices=train_indices)

        with self.profiler.profile("Load All Factors"):
            loaded_factors = self._load_all_required_factors(
                factors_name=factors_name,
                strategy=strategy,
                factor_config=factor_config,
            )

        no_partition = score_args.data_partition_size == 1 and score_args.module_partition_size == 1
        partition_provided = target_data_partitions is not None or target_module_partitions is not None

        if no_partition and partition_provided:
            error_msg = (
                "`target_data_partitions` or `target_module_partitions` were specified, while"
                "the `ScoreArguments` did not expect any partitions for computing influence scores."
            )
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        if no_partition:
            if per_device_train_batch_size is None:
                per_device_train_batch_size = self._find_executable_self_scores_batch_size(
                    train_dataset=train_dataset,
                    loaded_factors=loaded_factors,
                    dataloader_params=dataloader_params,
                    total_data_examples=len(train_dataset),
                    score_args=score_args,
                    factor_args=factor_args,
                    tracked_modules_name=None,
                )
            scores = self._fit_partitioned_self_scores(
                loaded_factors=loaded_factors,
                train_dataset=train_dataset,
                per_device_train_batch_size=per_device_train_batch_size,
                dataloader_params=dataloader_params,
                score_args=score_args,
                factor_args=factor_args,
                indices=None,
                tracked_module_names=None,
            )
            with self.profiler.profile("Save Self-Influence Score"):
                if self.state.is_main_process:
                    save_self_scores(
                        output_dir=scores_output_dir,
                        scores=scores,
                    )
                self.state.wait_for_everyone()
            self.logger.info(f"Saved self-influence scores at {scores_output_dir}.")

        else:
            data_partition_indices, target_data_partitions = self._get_data_partition(
                total_data_examples=len(train_dataset),
                data_partition_size=score_args.data_partition_size,
                target_data_partitions=target_data_partitions,
            )
            module_partition_names, target_module_partitions = self._get_module_partition(
                module_partition_size=score_args.module_partition_size,
                target_module_partitions=target_module_partitions,
            )

            all_start_time = get_time(state=self.state)
            for data_partition in target_data_partitions:
                for module_partition in target_module_partitions:
                    if (
                        self_scores_exist(
                            output_dir=scores_output_dir,
                            partition=(data_partition, module_partition),
                        )
                        and not overwrite_output_dir
                    ):
                        self.logger.info(
                            f"Found existing self-influence scores for data partition {data_partition} "
                            f"and module partition {module_partition} at {scores_output_dir}. Skipping."
                        )
                        continue

                    start_index, end_index = data_partition_indices[data_partition]
                    self.logger.info(
                        f"Computing self-influence scores for data partition with data indices ({start_index}, "
                        f"{end_index}) and modules {module_partition_names[module_partition]}..."
                    )

                    if per_device_train_batch_size is None:
                        per_device_train_batch_size = self._find_executable_self_scores_batch_size(
                            train_dataset=train_dataset,
                            loaded_factors=loaded_factors,
                            dataloader_params=dataloader_params,
                            total_data_examples=len(train_dataset) // score_args.data_partition_size,
                            score_args=score_args,
                            factor_args=factor_args,
                            tracked_modules_name=module_partition_names[0],
                        )
                    scores = self._fit_partitioned_self_scores(
                        loaded_factors=loaded_factors,
                        train_dataset=train_dataset,
                        per_device_train_batch_size=per_device_train_batch_size,
                        dataloader_params=dataloader_params,
                        score_args=score_args,
                        factor_args=factor_args,
                        indices=list(range(start_index, end_index)),
                        tracked_module_names=module_partition_names[module_partition],
                    )
                    with self.profiler.profile("Save Self-Influence Score"):
                        if self.state.is_main_process:
                            save_self_scores(
                                output_dir=scores_output_dir,
                                scores=scores,
                                partition=(data_partition, module_partition),
                            )
                        self.state.wait_for_everyone()
                    del scores
                    self.logger.info(f"Saved partitioned self-influence scores at {scores_output_dir}.")

            all_end_time = get_time(state=self.state)
            elapsed_time = all_end_time - all_start_time
            self.logger.info(f"Fitted all partitioned self-influence scores in {elapsed_time:.2f} seconds.")
            self.aggregate_self_scores(scores_name=scores_name, score_args=score_args)

        profile_summary = self.profiler.summary()
        if profile_summary != "":
            self.logger.info(self.profiler.summary())

    @torch.no_grad()
    def aggregate_self_scores(self, scores_name: str, score_args: ScoreArguments) -> None:
        """Aggregates self-influence scores computed for all data and module partitions."""
        self._aggregate_scores(
            scores_name=scores_name,
            score_args=score_args,
            exists_fnc=self_scores_exist,
            load_fnc=load_self_scores,
            save_fnc=save_self_scores,
            dim=0,
        )
