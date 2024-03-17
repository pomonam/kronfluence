import os
from typing import Callable, Optional, Sequence

import torch
from score.self import (
    compute_self_scores_with_loaders,
    load_self_scores,
    save_self_scores,
    self_scores_exist,
)
from torch.utils import data

from kronfluence.arguments import FactorArguments, ScoreArguments
from kronfluence.computer.computer import Computer
from kronfluence.module.constants import FACTOR_TYPE, SCORE_TYPE
from kronfluence.module.tracked_module import ModuleMode
from kronfluence.module.utils import set_mode
from kronfluence.score.pairwise import (
    compute_pairwise_scores_with_loaders,
    load_pairwise_scores,
    pairwise_scores_exist,
    save_pairwise_scores,
)
from kronfluence.utils.dataset import DataLoaderKwargs, find_executable_batch_size
from kronfluence.utils.exceptions import FactorsNotFoundError
from kronfluence.utils.logger import get_time
from kronfluence.utils.save import FACTOR_ARGUMENTS_NAME, SCORE_ARGUMENTS_NAME
from kronfluence.utils.state import release_memory


class ScoreComputer(Computer):
    """Handles the computation of pairwise influence scores for a given PyTorch model."""

    def _find_executable_scores_batch_size(
        self,
        func: Callable,
        factor_args: FactorArguments,
        loaded_factors,
        query_dataset: data.Dataset,
        dataloader_params,
        per_device_query_batch_size,
        train_dataset: data.Dataset,
        score_args,
        tracked_modules_name,
        total_data_examples: Optional[int] = None,
    ) -> int:
        """Automatically finds executable batch size for performing `func`."""
        if self.state.num_processes > 1:
            error_msg = (
                "Automatic batch size search is currently not supported for multi-GPU training. "
                "Please manually configure the batch size."
            )
            self.logger.error(error_msg)
            raise NotImplementedError(error_msg)

        self.logger.info("Automatically determining executable batch size.")

        if total_data_examples is None:
            total_data_examples = len(train_dataset)
        start_batch_size = min(
            [
                factor_args.initial_per_device_batch_size_attempt,
                total_data_examples,
            ]
        )

        total_query_batch_size = per_device_query_batch_size * self.state.num_processes
        query_dataset = data.Subset(dataset=query_dataset, indices=list(range(total_query_batch_size)))

        def executable_batch_size_func(batch_size: int) -> None:
            self.logger.info(f"Attempting to set per-device batch size to {batch_size}.")
            set_mode(model=self.model, mode=ModuleMode.DEFAULT, keep_factors=False)
            self.model.zero_grad(set_to_none=True)
            release_memory()
            total_batch_size = batch_size * self.state.num_processes
            func(
                loaded_factors=loaded_factors,
                query_dataset=query_dataset,
                train_dataset=train_dataset,
                per_device_query_batch_size=total_query_batch_size,
                per_device_train_batch_size=batch_size,
                dataloader_params=dataloader_params,
                score_args=score_args,
                factor_args=factor_args,
                indices=list(range(total_batch_size)),
                tracked_module_names=tracked_modules_name,
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
        query_indices: Optional[Sequence[int]] = None,
        train_indices: Optional[Sequence[int]] = None,
        dataloader_kwargs: Optional[DataLoaderKwargs] = None,
        score_args: Optional[ScoreArguments] = None,
        target_data_partitions: Optional[Sequence[int]] = None,
        target_module_partitions: Optional[Sequence[int]] = None,
        overwrite_output_dir: bool = False,
    ) -> None:
        """Fits all pairwise scores for the given data and module partition."""
        self.logger.debug(f"Computing pairwise scores with parameters: {locals()}")

        def compute_fnc(
            loaded_factors,
            query_dataset,
            per_device_query_batch_size,
            train_dataset,
            per_device_train_batch_size,
            dataloader_params,
            score_args,
            factor_args,
            indices,
            tracked_module_names,
        ) -> SCORE_TYPE:
            query_loader = self._get_dataloader(
                dataset=query_dataset,
                per_device_batch_size=per_device_query_batch_size,
                allow_duplicates=True,
                dataloader_params=dataloader_params,
            )
            train_loader = self._get_dataloader(
                dataset=train_dataset,
                per_device_batch_size=per_device_train_batch_size,
                indices=indices,
                allow_duplicates=True,
                stack=True,
                dataloader_params=dataloader_params,
            )
            scores = compute_pairwise_scores_with_loaders(
                model=self.model,
                state=self.state,
                task=self.task,
                loaded_factors=loaded_factors,
                query_loader=query_loader,
                train_loader=train_loader,
                per_device_query_batch_size=per_device_query_batch_size,
                score_args=score_args,
                factor_args=factor_args,
                tracked_module_names=tracked_module_names,
            )
            return scores

        self._compute_scores(
            scores_name=scores_name,
            factors_name=factors_name,
            query_dataset=query_dataset,
            train_dataset=train_dataset,
            exist_fnc=pairwise_scores_exist,
            compute_fnc=compute_fnc,
            save_fnc=save_pairwise_scores,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_query_batch_size=per_device_query_batch_size,
            query_indices=query_indices,
            train_indices=train_indices,
            dataloader_kwargs=dataloader_kwargs,
            score_args=score_args,
            target_data_partitions=target_data_partitions,
            target_module_partitions=target_module_partitions,
            overwrite_output_dir=overwrite_output_dir,
        )

        self.aggregate_pairwise_scores(scores_name)
        self._log_profile_summary()

    @torch.no_grad()
    def aggregate_pairwise_scores(self, scores_name: str) -> None:
        """Aggregates pairwise scores computed for all data and module partitions."""
        score_args = self._load_and_configure_score_args(scores_name=scores_name)
        no_partition = score_args.data_partition_size == 1 and score_args.module_partition_size == 1
        if not no_partition:
            self._aggregate_scores(
                scores_name=scores_name,
                score_args=score_args,
                exists_fnc=pairwise_scores_exist,
                load_fnc=load_pairwise_scores,
                save_fnc=save_pairwise_scores,
                dim=1,
            )

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
        """Fits all pairwise scores for the given data and module partition."""
        self.logger.debug(f"Computing pairwise scores with parameters: {locals()}")

        def compute_fnc(
            loaded_factors,
            query_dataset,
            per_device_query_batch_size,
            train_dataset,
            per_device_train_batch_size,
            dataloader_params,
            score_args,
            factor_args,
            indices,
            tracked_module_names,
        ) -> SCORE_TYPE:
            del query_dataset, per_device_query_batch_size
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
            return scores

        self._compute_scores(
            scores_name=scores_name,
            factors_name=factors_name,
            query_dataset=None,
            train_dataset=train_dataset,
            exist_fnc=self_scores_exist,
            compute_fnc=compute_fnc,
            save_fnc=save_self_scores,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_query_batch_size=None,
            query_indices=None,
            train_indices=train_indices,
            dataloader_kwargs=dataloader_kwargs,
            score_args=score_args,
            target_data_partitions=target_data_partitions,
            target_module_partitions=target_module_partitions,
            overwrite_output_dir=overwrite_output_dir,
        )

        self.aggregate_self_scores(scores_name)
        self._log_profile_summary()

    @torch.no_grad()
    def aggregate_self_scores(self, scores_name: str) -> None:
        """Aggregates pairwise scores computed for all data and module partitions."""
        score_args = self._load_and_configure_score_args(scores_name=scores_name)
        no_partition = score_args.data_partition_size == 1 and score_args.module_partition_size == 1
        if not no_partition:
            self._aggregate_scores(
                scores_name=scores_name,
                score_args=score_args,
                exists_fnc=self_scores_exist,
                load_fnc=load_self_scores,
                save_fnc=save_self_scores,
                dim=0,
            )

    def _compute_scores(
        self,
        scores_name: str,
        factors_name: str,
        query_dataset: Optional[data.Dataset],
        train_dataset: data.Dataset,
        exist_fnc,
        compute_fnc,
        save_fnc,
        per_device_query_batch_size: Optional[int],
        per_device_train_batch_size: Optional[int] = None,
        query_indices: Optional[Sequence[int]] = None,
        train_indices: Optional[Sequence[int]] = None,
        dataloader_kwargs: Optional[DataLoaderKwargs] = None,
        score_args: Optional[ScoreArguments] = None,
        target_data_partitions: Optional[Sequence[int]] = None,
        target_module_partitions: Optional[Sequence[int]] = None,
        overwrite_output_dir: bool = False,
    ) -> Optional[SCORE_TYPE]:
        scores_output_dir = self.scores_output_dir(scores_name=scores_name)
        os.makedirs(scores_output_dir, exist_ok=True)
        if exist_fnc(output_dir=scores_output_dir) and not overwrite_output_dir:
            self.logger.info(f"Found existing scores at {scores_output_dir}. Skipping.")
            return

        score_args = self._configure_score_args(score_args)
        factor_args, factor_config = self._load_and_configure_factor_args(factors_name=factors_name)

        if self.state.is_main_process:
            if query_dataset is not None:
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

        dataloader_params = self._configure_dataloader(dataloader_kwargs)
        if query_indices is not None:
            query_dataset = data.Subset(dataset=query_dataset, indices=query_indices)
        if train_indices is not None:
            train_dataset = data.Subset(dataset=train_dataset, indices=train_indices)

        with self.profiler.profile("Load All Factors"):
            loaded_factors = self._load_all_required_factors(
                factors_name=factors_name,
                strategy=factor_args.strategy,
                factor_config=factor_config,
            )

        no_partition = score_args.data_partition_size == 1 and score_args.module_partition_size == 1
        partition_provided = target_data_partitions is not None or target_module_partitions is not None
        if no_partition and partition_provided:
            error_msg = (
                "`target_data_partitions` or `target_module_partitions` were specified, while"
                "the `ScoreArguments` did not expect any data or module partition to compute scores."
            )
            self.logger.error(error_msg)
            raise ValueError(error_msg)

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
                if no_partition:
                    partition = None
                else:
                    partition = (data_partition, module_partition)

                if (
                    exist_fnc(
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
                    f"Computing pairwise scores for data partition with data indices ({start_index}, "
                    f"{end_index}) and modules {module_partition_names[module_partition]}..."
                )

                if per_device_train_batch_size is None:
                    per_device_train_batch_size = self._find_executable_scores_batch_size(
                        loaded_factors=loaded_factors,
                        func=compute_fnc,
                        query_dataset=query_dataset,
                        per_device_query_batch_size=per_device_query_batch_size,
                        train_dataset=train_dataset,
                        dataloader_params=dataloader_params,
                        total_data_examples=len(train_dataset) // score_args.data_partition_size,
                        score_args=score_args,
                        factor_args=factor_args,
                        tracked_modules_name=module_partition_names[0],
                    )

                release_memory()
                start_time = get_time(state=self.state)
                with self.profiler.profile("Compute Score"):
                    scores = compute_fnc(
                        loaded_factors=loaded_factors,
                        query_dataset=query_dataset,
                        per_device_query_batch_size=per_device_query_batch_size,
                        train_dataset=train_dataset,
                        per_device_train_batch_size=per_device_train_batch_size,
                        dataloader_params=dataloader_params,
                        score_args=score_args,
                        factor_args=factor_args,
                        indices=list(range(start_index, end_index)),
                        tracked_module_names=module_partition_names[module_partition],
                    )
                end_time = get_time(state=self.state)
                elapsed_time = end_time - start_time
                self.logger.info(f"Computed pairwise scores in {elapsed_time:.2f} seconds.")

                with self.profiler.profile("Save Score"):
                    if self.state.is_main_process:
                        save_fnc(
                            output_dir=scores_output_dir,
                            scores=scores,
                            partition=partition,
                        )
                    self.state.wait_for_everyone()
                del scores
                self.logger.info(f"Saved partitioned pairwise scores at {scores_output_dir}.")

        all_end_time = get_time(state=self.state)
        elapsed_time = all_end_time - all_start_time
        if not no_partition:
            self.logger.info(f"Computed all scores in {elapsed_time:.2f} seconds.")
