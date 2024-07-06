import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils import data

from kronfluence.arguments import FactorArguments, ScoreArguments
from kronfluence.computer.computer import Computer
from kronfluence.score.pairwise import (
    compute_pairwise_query_aggregated_scores_with_loaders,
    compute_pairwise_scores_with_loaders,
    load_pairwise_scores,
    pairwise_scores_exist,
    save_pairwise_scores,
)
from kronfluence.score.self import (
    compute_self_measurement_scores_with_loaders,
    compute_self_scores_with_loaders,
    load_self_scores,
    save_self_scores,
    self_scores_exist,
)
from kronfluence.utils.constants import (
    FACTOR_ARGUMENTS_NAME,
    FACTOR_TYPE,
    SCORE_ARGUMENTS_NAME,
    SCORE_TYPE,
)
from kronfluence.utils.dataset import DataLoaderKwargs, find_executable_batch_size
from kronfluence.utils.exceptions import FactorsNotFoundError
from kronfluence.utils.logger import get_time


class ScoreComputer(Computer):
    """Handles the computation of influence scores for a given PyTorch model."""

    def _configure_and_save_score_args(
        self,
        score_args: Optional[FactorArguments],
        scores_output_dir: Path,
        factors_name: str,
        overwrite_output_dir: bool,
    ) -> Tuple[FactorArguments, ScoreArguments]:
        """Configures and saves score arguments to disk."""
        if score_args is None:
            score_args = ScoreArguments()
            self.logger.info(f"Score arguments not provided. Using the default configuration: {score_args}.")
        else:
            self.logger.info(f"Using the provided configuration: {score_args}.")

        factor_args = self.load_factor_args(factors_name=factors_name)
        factors_output_dir = self.factors_output_dir(factors_name=factors_name)
        if factor_args is None:
            error_msg = f"Factors with name `{factors_name}` not found at `{factors_output_dir}`."
            self.logger.error(error_msg)
            raise FactorsNotFoundError(error_msg)
        self.logger.info(f"Loaded `FactorArguments` with configuration: {factor_args}.")

        if self.state.is_main_process:
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
        self.state.wait_for_everyone()
        return factor_args, score_args

    @torch.no_grad()
    def _aggregate_scores(
        self,
        scores_name: str,
        score_args: ScoreArguments,
        exist_fnc: Callable,
        load_fnc: Callable,
        save_fnc: Callable,
        dim: int,
    ) -> Optional[SCORE_TYPE]:
        """Aggregates influence scores computed for all data and module partitions."""
        scores_output_dir = self.scores_output_dir(scores_name=scores_name)
        if not scores_output_dir.exists():
            error_msg = f"Scores directory `{scores_output_dir}` not found when trying to aggregate scores."
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        all_required_partitions = [
            (i, j) for i in range(score_args.data_partitions) for j in range(score_args.module_partitions)
        ]
        all_partition_exists = all(
            exist_fnc(output_dir=scores_output_dir, partition=partition) for partition in all_required_partitions
        )
        if not all_partition_exists:
            self.logger.info("Scores are not aggregated as scores for some partitions are not yet computed.")
            return

        start_time = time.time()
        aggregated_scores: SCORE_TYPE = {}
        for data_partition in range(score_args.data_partitions):
            aggregated_module_scores = {}

            for module_partition in range(score_args.module_partitions):
                loaded_scores = load_fnc(
                    output_dir=scores_output_dir,
                    partition=(data_partition, module_partition),
                )

                for module_name, scores in loaded_scores.items():
                    if module_name not in aggregated_module_scores:
                        aggregated_module_scores[module_name] = torch.zeros_like(scores, requires_grad=False)
                    aggregated_module_scores[module_name].add_(scores)
                del loaded_scores

            for module_name, scores in aggregated_module_scores.items():
                if module_name not in aggregated_scores:
                    aggregated_scores[module_name] = scores.clone()
                else:
                    if score_args.aggregate_train_gradients:
                        aggregated_scores[module_name].add_(scores)
                    else:
                        aggregated_scores[module_name] = torch.cat(
                            (
                                aggregated_scores[module_name],
                                scores,
                            ),
                            dim=dim,
                        )
        save_fnc(output_dir=scores_output_dir, scores=aggregated_scores, metadata=score_args.to_str_dict())
        end_time = time.time()
        elapsed_time = end_time - start_time
        self.logger.info(f"Aggregated all scores in {elapsed_time:.2f} seconds.")
        return aggregated_scores

    def _find_executable_pairwise_scores_batch_size(
        self,
        loaded_factors: FACTOR_TYPE,
        query_dataset: data.Dataset,
        per_device_query_batch_size: int,
        train_dataset: data.Dataset,
        initial_per_device_train_batch_size_attempt: int,
        total_data_examples: int,
        dataloader_params: Dict[str, Any],
        score_args: ScoreArguments,
        factor_args: FactorArguments,
        tracked_modules_name: Optional[List[str]],
    ) -> int:
        """Automatically finds executable training batch size for computing pairwise influence scores."""
        if self.state.use_distributed:
            error_msg = (
                "Automatic batch size search is not supported for multi-GPU setting. "
                "Please manually configure the batch size by passing in `per_device_batch_size`."
            )
            self.logger.error(error_msg)
            raise NotImplementedError(error_msg)

        self.logger.info("Automatically determining executable batch size.")
        total_query_batch_size = per_device_query_batch_size * self.state.num_processes
        start_batch_size = min(
            [
                initial_per_device_train_batch_size_attempt,
                total_data_examples,
            ]
        )

        def executable_batch_size_func(batch_size: int) -> None:
            self.logger.info(f"Attempting to set per-device batch size to {batch_size}.")
            # Releases all memory that could be caused by the previous OOM.
            self._reset_memory()
            total_batch_size = batch_size * self.state.num_processes
            query_loader = self._get_dataloader(
                dataset=query_dataset,
                per_device_batch_size=per_device_query_batch_size,
                indices=list(range(total_query_batch_size)),
                dataloader_params=dataloader_params,
                allow_duplicates=True,
            )
            train_loader = self._get_dataloader(
                dataset=train_dataset,
                per_device_batch_size=batch_size,
                indices=list(range(total_batch_size)),
                dataloader_params=dataloader_params,
                allow_duplicates=True,
                stack=True,
            )
            func = (
                compute_pairwise_scores_with_loaders
                if not score_args.aggregate_query_gradients
                else compute_pairwise_query_aggregated_scores_with_loaders
            )
            func(
                model=self.model,
                state=self.state,
                task=self.task,
                loaded_factors=loaded_factors,
                score_args=score_args,
                factor_args=factor_args,
                query_loader=query_loader,
                train_loader=train_loader,
                per_device_query_batch_size=per_device_query_batch_size,
                tracked_module_names=tracked_modules_name,
                disable_tqdm=True,
            )

        per_device_batch_size = find_executable_batch_size(
            func=executable_batch_size_func,
            start_batch_size=start_batch_size,
        )
        self.logger.info(f"Executable batch size determined: {per_device_batch_size}.")
        return per_device_batch_size

    def compute_pairwise_scores(
        self,
        scores_name: str,
        factors_name: str,
        query_dataset: data.Dataset,
        train_dataset: data.Dataset,
        per_device_query_batch_size: int,
        per_device_train_batch_size: Optional[int] = None,
        initial_per_device_train_batch_size_attempt: int = 4096,
        query_indices: Optional[Sequence[int]] = None,
        train_indices: Optional[Sequence[int]] = None,
        dataloader_kwargs: Optional[DataLoaderKwargs] = None,
        score_args: Optional[ScoreArguments] = None,
        target_data_partitions: Optional[Sequence[int]] = None,
        target_module_partitions: Optional[Sequence[int]] = None,
        overwrite_output_dir: bool = False,
    ) -> Optional[SCORE_TYPE]:
        """Computes pairwise influence scores with the given score configuration.

        Args:
            scores_name (str):
                The unique identifier for the score, used to organize and retrieve the results.
            factors_name (str):
                The name of the factor to use for influence computations.
            query_dataset (data.Dataset):
                The query dataset, typically much smaller than the training dataset.
            train_dataset (data.Dataset):
                The training dataset.
            per_device_query_batch_size (int):
                The per-device batch size used to compute query gradients.
            per_device_train_batch_size (int, optional):
                The per-device batch size used to compute training gradients. If not specified, an executable
                batch size will be found.
            initial_per_device_train_batch_size_attempt (int, optional):
                The initial attempted per-device batch size when the batch size is not provided.
            query_indices (Sequence[int], optional):
                The specific indices of the query dataset to compute the influence scores for. If not specified,
                all query data points will be used.
            train_indices (Sequence[int], optional):
                The specific indices of the training dataset to compute the influence scores for. If not
                specified, all training data points will be used.
            dataloader_kwargs (DataLoaderKwargs, optional):
                Controls additional arguments for PyTorch's DataLoader.
            score_args (ScoreArguments, optional):
                Arguments for score computation.
            target_data_partitions (Sequence[int], optional):
                Specific data partitions to compute influence scores. If not specified, scores for all
                data partitions will be computed.
            target_module_partitions (Sequence[int], optional):
                Specific module partitions to compute influence scores. If not specified, scores for all
                module partitions will be computed.
            overwrite_output_dir (bool, optional):
                Whether to overwrite existing output.
        """
        self.logger.debug(f"Computing pairwise scores with parameters: {locals()}")

        scores_output_dir = self.scores_output_dir(scores_name=scores_name)
        os.makedirs(scores_output_dir, exist_ok=True)
        if pairwise_scores_exist(output_dir=scores_output_dir) and not overwrite_output_dir:
            self.logger.info(f"Found existing pairwise scores at `{scores_output_dir}`. Skipping.")
            return self.load_pairwise_scores(scores_name=scores_name)

        factor_args, score_args = self._configure_and_save_score_args(
            score_args=score_args,
            scores_output_dir=scores_output_dir,
            factors_name=factors_name,
            overwrite_output_dir=overwrite_output_dir,
        )

        if score_args.compute_per_token_scores and score_args.aggregate_train_gradients:
            warning_msg = (
                "Token-wise influence computation is not compatible with `aggregate_train_gradients=True`. "
                "Disabling `compute_per_token_scores`."
            )
            score_args.compute_per_token_scores = False
            self.logger.warning(warning_msg)

        if score_args.compute_per_token_scores and factor_args.has_shared_parameters:
            warning_msg = (
                "Token-wise influence computation is not compatible with `has_shared_parameters=True`. "
                "Disabling `compute_per_token_scores`."
            )
            score_args.compute_per_token_scores = False
            self.logger.warning(warning_msg)

        if score_args.compute_per_token_scores and self.task.enable_post_process_per_sample_gradient:
            warning_msg = (
                "Token-wise influence computation is not compatible with tasks that requires "
                "`enable_post_process_per_sample_gradient`. Disabling `compute_per_token_scores`."
            )
            score_args.compute_per_token_scores = False
            self.logger.warning(warning_msg)

        dataloader_params = self._configure_dataloader(dataloader_kwargs)
        if self.state.is_main_process:
            self._save_dataset_metadata(
                dataset_name="query",
                dataset=query_dataset,
                indices=query_indices,
                output_dir=scores_output_dir,
                overwrite_output_dir=overwrite_output_dir,
            )
            self._save_dataset_metadata(
                dataset_name="train",
                dataset=train_dataset,
                indices=train_indices,
                output_dir=scores_output_dir,
                overwrite_output_dir=overwrite_output_dir,
            )
        if query_indices is not None:
            query_dataset = data.Subset(dataset=query_dataset, indices=query_indices)
            del query_indices

        if train_indices is not None:
            train_dataset = data.Subset(dataset=train_dataset, indices=train_indices)
            del train_indices

        with self.profiler.profile("Load All Factors"):
            loaded_factors = self.load_all_factors(
                factors_name=factors_name,
            )

        no_partition = score_args.data_partitions == 1 and score_args.module_partitions == 1
        partition_provided = target_data_partitions is not None or target_module_partitions is not None
        if no_partition and partition_provided:
            error_msg = (
                "`target_data_partitions` or `target_module_partitions` were specified, while"
                "the `ScoreArguments` did not expect any data and module partition to compute pairwise scores."
            )
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        data_partition_indices, target_data_partitions = self._get_data_partition(
            total_data_examples=len(train_dataset),
            data_partitions=score_args.data_partitions,
            target_data_partitions=target_data_partitions,
        )
        max_partition_examples = len(train_dataset) // score_args.data_partitions
        module_partition_names, target_module_partitions = self._get_module_partition(
            module_partitions=score_args.module_partitions,
            target_module_partitions=target_module_partitions,
        )

        all_start_time = get_time(state=self.state)
        for data_partition in target_data_partitions:
            for module_partition in target_module_partitions:
                if no_partition:
                    partition = None
                else:
                    partition = (data_partition, module_partition)

                if (
                    pairwise_scores_exist(
                        output_dir=scores_output_dir,
                        partition=partition,
                    )
                    and not overwrite_output_dir
                ):
                    self.logger.info(
                        f"Found existing pairwise scores for data partition {data_partition} "
                        f"and module partition {module_partition} at {scores_output_dir}. Skipping."
                    )
                    continue

                start_index, end_index = data_partition_indices[data_partition]
                self.logger.info(
                    f"Computing pairwise scores with data indices ({start_index}, {end_index}) and "
                    f"modules {module_partition_names[module_partition]}."
                )

                if per_device_train_batch_size is None:
                    per_device_train_batch_size = self._find_executable_pairwise_scores_batch_size(
                        query_dataset=query_dataset,
                        per_device_query_batch_size=per_device_query_batch_size
                        if not score_args.aggregate_query_gradients
                        else 1,
                        train_dataset=train_dataset,
                        initial_per_device_train_batch_size_attempt=initial_per_device_train_batch_size_attempt,
                        loaded_factors=loaded_factors,
                        dataloader_params=dataloader_params,
                        total_data_examples=max_partition_examples,
                        score_args=score_args,
                        factor_args=factor_args,
                        tracked_modules_name=module_partition_names[module_partition],
                    )

                self._reset_memory()
                start_time = get_time(state=self.state)
                with self.profiler.profile("Compute Pairwise Score"):
                    query_loader = self._get_dataloader(
                        dataset=query_dataset,
                        per_device_batch_size=per_device_query_batch_size,
                        dataloader_params=dataloader_params,
                        allow_duplicates=not score_args.aggregate_query_gradients,
                    )
                    train_loader = self._get_dataloader(
                        dataset=train_dataset,
                        per_device_batch_size=per_device_train_batch_size,
                        indices=list(range(start_index, end_index)),
                        dataloader_params=dataloader_params,
                        allow_duplicates=not score_args.aggregate_train_gradients,
                        stack=not score_args.aggregate_train_gradients,
                    )
                    func = (
                        compute_pairwise_scores_with_loaders
                        if not score_args.aggregate_query_gradients
                        else compute_pairwise_query_aggregated_scores_with_loaders
                    )
                    scores = func(
                        model=self.model,
                        state=self.state,
                        task=self.task,
                        loaded_factors=loaded_factors,
                        query_loader=query_loader,
                        train_loader=train_loader,
                        per_device_query_batch_size=per_device_query_batch_size,
                        score_args=score_args,
                        factor_args=factor_args,
                        tracked_module_names=module_partition_names[module_partition],
                        disable_tqdm=self.disable_tqdm,
                    )
                end_time = get_time(state=self.state)
                elapsed_time = end_time - start_time
                self.logger.info(f"Computed pairwise influence scores in {elapsed_time:.2f} seconds.")

                with self.profiler.profile("Save Pairwise Score"):
                    if self.state.is_main_process:
                        save_pairwise_scores(
                            output_dir=scores_output_dir,
                            scores=scores,
                            partition=partition,
                            metadata=score_args.to_str_dict(),
                        )
                    self.state.wait_for_everyone()
                del scores, query_loader, train_loader
                self._reset_memory()
                self.logger.info(f"Saved pairwise scores at {scores_output_dir}.")

        all_end_time = get_time(state=self.state)
        elapsed_time = all_end_time - all_start_time
        if not no_partition:
            self.logger.info(f"Fitted all partitioned pairwise scores in {elapsed_time:.2f} seconds.")
            if self.state.is_main_process:
                self.aggregate_pairwise_scores(scores_name=scores_name)
                self.logger.info(f"Saved aggregated pairwise scores at `{scores_output_dir}`.")
            self.state.wait_for_everyone()
        self._log_profile_summary(name=f"scores_{scores_name}_pairwise")

    @torch.no_grad()
    def aggregate_pairwise_scores(self, scores_name: str) -> None:
        """Aggregates all partitioned pairwise scores. The scores will not be aggregated if scores
        for some data or module partitions are missing.

        Args:
            scores_name (str):
                The unique identifier for the score, used to organize and retrieve the results.
        """
        score_args = self.load_score_args(scores_name=scores_name)
        if score_args is None:
            error_msg = (
                f"Arguments for scores with name `{score_args}` was not found when trying "
                f"to aggregate pairwise influence scores."
            )
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        with self.profiler.profile("Aggregate Score"):
            self._aggregate_scores(
                scores_name=scores_name,
                score_args=score_args,
                exist_fnc=pairwise_scores_exist,
                load_fnc=load_pairwise_scores,
                save_fnc=save_pairwise_scores,
                dim=1,
            )

    def _find_executable_self_scores_batch_size(
        self,
        loaded_factors: FACTOR_TYPE,
        train_dataset: data.Dataset,
        total_data_examples: int,
        initial_per_device_train_batch_size_attempt: int,
        dataloader_params: Dict[str, Any],
        score_args: ScoreArguments,
        factor_args: FactorArguments,
        tracked_modules_name: Optional[List[str]],
    ) -> int:
        """Automatically finds executable training batch size for computing self-influence scores."""
        if self.state.use_distributed:
            error_msg = (
                "Automatic batch size search is not supported for multi-GPU setting. "
                "Please manually configure the batch size by passing in `per_device_batch_size`."
            )
            self.logger.error(error_msg)
            raise NotImplementedError(error_msg)

        self.logger.info("Automatically determining executable batch size.")
        start_batch_size = min(
            [
                initial_per_device_train_batch_size_attempt,
                total_data_examples,
            ]
        )

        def executable_batch_size_func(batch_size: int) -> None:
            self.logger.info(f"Attempting to set per-device batch size to {batch_size}.")
            # Releases all memory that could be caused by the previous OOM.
            self._reset_memory()
            total_batch_size = batch_size * self.state.num_processes
            train_loader = self._get_dataloader(
                dataset=train_dataset,
                per_device_batch_size=batch_size,
                indices=list(range(total_batch_size)),
                dataloader_params=dataloader_params,
                allow_duplicates=True,
                stack=True,
            )
            if score_args.use_measurement_for_self_influence:
                func = compute_self_measurement_scores_with_loaders
            else:
                func = compute_self_scores_with_loaders
            func(
                model=self.model,
                state=self.state,
                task=self.task,
                loaded_factors=loaded_factors,
                train_loader=train_loader,
                score_args=score_args,
                factor_args=factor_args,
                tracked_module_names=tracked_modules_name,
                disable_tqdm=True,
            )

        per_device_batch_size = find_executable_batch_size(
            func=executable_batch_size_func,
            start_batch_size=start_batch_size,
        )
        self.logger.info(f"Executable batch size determined: {per_device_batch_size}.")
        return per_device_batch_size

    def compute_self_scores(
        self,
        scores_name: str,
        factors_name: str,
        train_dataset: data.Dataset,
        per_device_train_batch_size: Optional[int] = None,
        initial_per_device_train_batch_size_attempt: int = 4096,
        train_indices: Optional[Sequence[int]] = None,
        dataloader_kwargs: Optional[DataLoaderKwargs] = None,
        score_args: Optional[ScoreArguments] = None,
        target_data_partitions: Optional[Sequence[int]] = None,
        target_module_partitions: Optional[Sequence[int]] = None,
        overwrite_output_dir: bool = False,
    ) -> Optional[SCORE_TYPE]:
        """Computes self-influence scores with the given score configuration.

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
            initial_per_device_train_batch_size_attempt (int, optional):
                The initial attempted per-device batch size when the batch size is not provided.
            train_indices (Sequence[int], optional):
                The specific indices of the training dataset to compute the influence scores for. If not
                specified, all training data points will be used.
            dataloader_kwargs (DataLoaderKwargs, optional):
                Controls additional arguments for PyTorch's DataLoader.
            score_args (ScoreArguments, optional):
                Arguments for score computation.
            target_data_partitions (Sequence[int], optional):
                Specific data partitions to compute influence scores. If not specified, scores for all
                data partitions will be computed.
            target_module_partitions (Sequence[int], optional):
                Specific module partitions to compute influence scores. If not specified, scores for all
                module partitions will be computed.
            overwrite_output_dir (bool, optional):
                Whether to overwrite existing output.
        """
        self.logger.debug(f"Computing self-influence scores with parameters: {locals()}")

        scores_output_dir = self.scores_output_dir(scores_name=scores_name)
        os.makedirs(scores_output_dir, exist_ok=True)
        if self_scores_exist(output_dir=scores_output_dir) and not overwrite_output_dir:
            self.logger.info(f"Found existing self-influence scores at {scores_output_dir}. Skipping.")
            return self.load_self_scores(scores_name=scores_name)

        factor_args, score_args = self._configure_and_save_score_args(
            score_args=score_args,
            scores_output_dir=scores_output_dir,
            factors_name=factors_name,
            overwrite_output_dir=overwrite_output_dir,
        )

        if score_args.query_gradient_accumulation_steps != 1:
            warning_msg = "Query gradient accumulation is not supported for self-influence computation."
            score_args.query_gradient_accumulation_steps = 1
            self.logger.warning(warning_msg)

        if score_args.query_gradient_low_rank is not None:
            warning_msg = (
                "Low rank query gradient approximation is not supported for self-influence computation. "
                "No low rank query approximation will be performed."
            )
            score_args.query_gradient_low_rank = None
            self.logger.warning(warning_msg)

        if score_args.aggregate_query_gradients or score_args.aggregate_train_gradients:
            warning_msg = "Query or train gradient aggregation is not supported for self-influence computation."
            score_args.aggregate_train_gradients = False
            score_args.aggregate_query_gradients = False
            self.logger.warning(warning_msg)

        if score_args.compute_per_token_scores:
            warning_msg = "Token-wise influence computation is not compatible with self-influence scores. "
            score_args.compute_per_token_scores = False
            self.logger.warning(warning_msg)

        dataloader_params = self._configure_dataloader(dataloader_kwargs)
        if self.state.is_main_process:
            self._save_dataset_metadata(
                dataset_name="train",
                dataset=train_dataset,
                indices=train_indices,
                output_dir=scores_output_dir,
                overwrite_output_dir=overwrite_output_dir,
            )
        if train_indices is not None:
            train_dataset = data.Subset(dataset=train_dataset, indices=train_indices)
            del train_indices

        with self.profiler.profile("Load All Factors"):
            loaded_factors = self.load_all_factors(
                factors_name=factors_name,
            )

        no_partition = score_args.data_partitions == 1 and score_args.module_partitions == 1
        partition_provided = target_data_partitions is not None or target_module_partitions is not None
        if no_partition and partition_provided:
            error_msg = (
                "`target_data_partitions` or `target_module_partitions` were specified, while"
                "the `ScoreArguments` did not expect any data and module partition to compute self-influence scores."
            )
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        data_partition_indices, target_data_partitions = self._get_data_partition(
            total_data_examples=len(train_dataset),
            data_partitions=score_args.data_partitions,
            target_data_partitions=target_data_partitions,
        )
        max_partition_examples = len(train_dataset) // score_args.data_partitions
        module_partition_names, target_module_partitions = self._get_module_partition(
            module_partitions=score_args.module_partitions,
            target_module_partitions=target_module_partitions,
        )

        all_start_time = get_time(state=self.state)
        for data_partition in target_data_partitions:
            for module_partition in target_module_partitions:
                if no_partition:
                    partition = None
                else:
                    partition = (data_partition, module_partition)

                if (
                    self_scores_exist(
                        output_dir=scores_output_dir,
                        partition=partition,
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
                    f"Computing self-influence scores with data indices ({start_index}, {end_index}) and "
                    f"modules {module_partition_names[module_partition]}."
                )

                if per_device_train_batch_size is None:
                    per_device_train_batch_size = self._find_executable_self_scores_batch_size(
                        train_dataset=train_dataset,
                        loaded_factors=loaded_factors,
                        dataloader_params=dataloader_params,
                        total_data_examples=max_partition_examples,
                        initial_per_device_train_batch_size_attempt=initial_per_device_train_batch_size_attempt,
                        score_args=score_args,
                        factor_args=factor_args,
                        tracked_modules_name=module_partition_names[module_partition],
                    )

                self._reset_memory()
                start_time = get_time(state=self.state)
                with self.profiler.profile("Compute Self-Influence Score"):
                    train_loader = self._get_dataloader(
                        dataset=train_dataset,
                        per_device_batch_size=per_device_train_batch_size,
                        indices=list(range(start_index, end_index)),
                        dataloader_params=dataloader_params,
                        allow_duplicates=True,
                        stack=True,
                    )
                    if score_args.use_measurement_for_self_influence:
                        func = compute_self_measurement_scores_with_loaders
                    else:
                        func = compute_self_scores_with_loaders
                    scores = func(
                        model=self.model,
                        state=self.state,
                        task=self.task,
                        loaded_factors=loaded_factors,
                        train_loader=train_loader,
                        score_args=score_args,
                        factor_args=factor_args,
                        tracked_module_names=module_partition_names[module_partition],
                        disable_tqdm=self.disable_tqdm,
                    )
                end_time = get_time(state=self.state)
                elapsed_time = end_time - start_time
                self.logger.info(f"Computed self-influence scores in {elapsed_time:.2f} seconds.")

                with self.profiler.profile("Save Self-Influence Score"):
                    if self.state.is_main_process:
                        save_self_scores(
                            output_dir=scores_output_dir,
                            scores=scores,
                            partition=partition,
                            metadata=score_args.to_str_dict(),
                        )
                    self.state.wait_for_everyone()
                del scores, train_loader
                self._reset_memory()
                self.logger.info(f"Saved self-influence scores at `{scores_output_dir}`.")

        all_end_time = get_time(state=self.state)
        elapsed_time = all_end_time - all_start_time
        if not no_partition:
            self.logger.info(f"Fitted all partitioned self-influence scores in {elapsed_time:.2f} seconds.")
            if self.state.is_main_process:
                self.aggregate_self_scores(scores_name=scores_name)
                self.logger.info(f"Saved aggregated self-influence scores at `{scores_output_dir}`.")
            self.state.wait_for_everyone()
        self._log_profile_summary(name=f"scores_{scores_name}_self")

    @torch.no_grad()
    def aggregate_self_scores(self, scores_name: str) -> None:
        """Aggregates all partitioned self-influence scores. The scores will not be aggregated if scores
        for some data or module partitions are missing.

        Args:
            scores_name (str):
                The unique identifier for the score, used to organize and retrieve the results.
        """
        score_args = self.load_score_args(scores_name=scores_name)
        if score_args is None:
            error_msg = (
                f"Arguments for scores with name `{score_args}` was not found when trying "
                f"to aggregate self-influence scores."
            )
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        score_args.aggregate_query_gradients = score_args.aggregate_train_gradients = False
        self._aggregate_scores(
            scores_name=scores_name,
            score_args=score_args,
            exist_fnc=self_scores_exist,
            load_fnc=load_self_scores,
            save_fnc=save_self_scores,
            dim=0,
        )
