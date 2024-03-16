import os
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch.utils import data

from kronfluence.arguments import FactorArguments
from kronfluence.computer.computer import Computer
from kronfluence.factor.config import FactorConfig
from kronfluence.factor.covariance import (
    covariance_matrices_exist,
    fit_covariance_matrices_with_loader,
    load_covariance_matrices,
    save_covariance_matrices,
)
from kronfluence.module.constants import FACTOR_TYPE
from kronfluence.utils.dataset import DataLoaderKwargs
from kronfluence.utils.logger import get_time
from kronfluence.utils.save import FACTOR_ARGUMENTS_NAME
from kronfluence.utils.state import release_memory


class CovarianceComputer(Computer):
    """Handles the computation of all covariance matrices for a given PyTorch model."""

    def _find_executable_covariance_factors_batch_size(
        self,
        total_data_examples: int,
        dataset: data.Dataset,
        dataloader_params: Dict[str, Any],
        factor_args: FactorArguments,
        tracked_module_names: Optional[List[str]],
    ) -> int:
        """Automatically finds executable batch size for computing covariance matrices."""
        if self.state.num_processes > 1:
            error_msg = (
                "Automatic batch size search is currently not supported for multi-GPU training. "
                "Please manually configure the batch size."
            )
            self.logger.error(error_msg)
            raise NotImplementedError(error_msg)

        kwargs = {
            "model": self.model,
            "state": self.state,
            "task": self.task,
            "factor_args": factor_args,
            "tracked_module_names": tracked_module_names,
        }
        start_batch_size = min(
            [
                factor_args.initial_per_device_batch_size_attempt,
                total_data_examples,
            ]
        )
        return self._find_executable_factors_batch_size(
            func=fit_covariance_matrices_with_loader,
            func_kwargs=kwargs,
            dataset=dataset,
            dataloader_params=dataloader_params,
            start_batch_size=start_batch_size,
        )

    def _fit_partitioned_covariance_matrices(
        self,
        dataset: data.Dataset,
        per_device_batch_size: int,
        dataloader_params: Dict[str, Any],
        factor_args: FactorArguments,
        indices: Optional[List[int]] = None,
        tracked_module_names: Optional[List[str]] = None,
    ) -> FACTOR_TYPE:
        """Fits all covariance matrices for the given data and module partition."""
        release_memory()
        start_time = get_time(state=self.state)
        with self.profiler.profile("Fit Covariance"):
            loader = self._get_dataloader(
                dataset=dataset,
                per_device_batch_size=per_device_batch_size,
                dataloader_params=dataloader_params,
                indices=indices,
            )
            num_data_processed, covariance_factors = fit_covariance_matrices_with_loader(
                model=self.model,
                state=self.state,
                task=self.task,
                loader=loader,
                factor_args=factor_args,
                tracked_module_names=tracked_module_names,
            )
        end_time = get_time(state=self.state)
        elapsed_time = end_time - start_time
        self.logger.info(
            f"Fitted covariance matrices on {num_data_processed.item()} data points in " f"{elapsed_time:.2f} seconds."
        )
        return covariance_factors

    def fit_covariance_matrices(
        self,
        factors_name: str,
        dataset: data.Dataset,
        per_device_batch_size: Optional[int] = None,
        dataloader_kwargs: Optional[DataLoaderKwargs] = None,
        factor_args: Optional[FactorArguments] = None,
        target_data_partitions: Optional[Sequence[int]] = None,
        target_module_partitions: Optional[Sequence[int]] = None,
        overwrite_output_dir: bool = False,
    ) -> Optional[FACTOR_TYPE]:
        """Computes covariance matrices for all available modules. See `fit_all_factors` for
        the complete docstring with detailed description of each parameter."""
        self.logger.debug(f"Fitting covariance matrices with parameters: {locals()}")

        factors_output_dir = self.factors_output_dir(factors_name=factors_name)
        os.makedirs(factors_output_dir, exist_ok=True)
        if covariance_matrices_exist(output_dir=factors_output_dir) and not overwrite_output_dir:
            self.logger.info(f"Found existing covariance matrices at {factors_output_dir}. Skipping.")
            return self.load_covariance_matrices(factors_name=factors_name)

        if factor_args is None:
            factor_args = FactorArguments()
            self.logger.info(f"Factor arguments not provided. Using the default configuration: {factor_args}.")
        else:
            self.logger.info(f"Using the provided configuration: {factor_args}.")

        if self.state.is_main_process:
            self._save_arguments(
                arguments_name=FACTOR_ARGUMENTS_NAME,
                arguments=factor_args,
                output_dir=factors_output_dir,
                overwrite_output_dir=overwrite_output_dir,
            )

        if not FactorConfig.CONFIGS[factor_args.strategy].requires_covariance_matrices:
            self.logger.info(
                f"Strategy `{factor_args.strategy}` does not require fitting covariance matrices. " f"Skipping."
            )
            return

        if self.state.is_main_process:
            self._save_dataset_metadata(
                dataset_name="covariance",
                dataset=dataset,
                output_dir=factors_output_dir,
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

        total_data_examples = min([factor_args.covariance_max_examples, len(dataset)])
        self.logger.info(f"Total data examples to fit covariance matrices: {total_data_examples}.")

        no_partition = (
            factor_args.covariance_data_partition_size == 1 and factor_args.covariance_module_partition_size == 1
        )
        partition_provided = target_data_partitions is not None or target_module_partitions is not None
        if no_partition and partition_provided:
            error_msg = (
                "`target_data_partitions` or `target_module_partitions` were specified, while"
                "the `FactorArguments` did not expect any partitions when computing covariance matrices."
            )
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        if no_partition:
            if total_data_examples < self.state.num_processes:
                error_msg = "The number of processes are more than the data examples."
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            if per_device_batch_size is None:
                per_device_batch_size = self._find_executable_covariance_factors_batch_size(
                    dataloader_params=dataloader_params,
                    dataset=dataset,
                    total_data_examples=total_data_examples,
                    factor_args=factor_args,
                    tracked_module_names=None,
                )
            covariance_factors = self._fit_partitioned_covariance_matrices(
                dataset=dataset,
                per_device_batch_size=per_device_batch_size,
                dataloader_params=dataloader_params,
                factor_args=factor_args,
                indices=list(range(total_data_examples)),
                tracked_module_names=None,
            )
            with self.profiler.profile("Save Covariance"):
                if self.state.is_main_process:
                    save_covariance_matrices(
                        output_dir=factors_output_dir,
                        covariance_factors=covariance_factors,
                    )
                self.state.wait_for_everyone()
            self.logger.info(f"Saved covariance matrices at {factors_output_dir}.")

            profile_summary = self.profiler.summary()
            if profile_summary != "":
                self.logger.info(self.profiler.summary())
            return covariance_factors

        data_partition_indices, target_data_partitions = self._get_data_partition(
            total_data_examples=total_data_examples,
            data_partition_size=factor_args.covariance_data_partition_size,
            target_data_partitions=target_data_partitions,
        )
        module_partition_names, target_module_partitions = self._get_module_partition(
            module_partition_size=factor_args.covariance_module_partition_size,
            target_module_partitions=target_module_partitions,
        )

        all_start_time = get_time(state=self.state)
        for data_partition in target_data_partitions:
            for module_partition in target_module_partitions:
                if (
                    covariance_matrices_exist(
                        output_dir=factors_output_dir,
                        partition=(data_partition, module_partition),
                    )
                    and not overwrite_output_dir
                ):
                    self.logger.info(
                        f"Found existing covariance matrices for data partition {data_partition} "
                        f"and module partition {module_partition} at {factors_output_dir}. Skipping."
                    )
                    continue

                start_index, end_index = data_partition_indices[data_partition]
                self.logger.info(
                    f"Fitting covariance matrices for data partition with data indices ({start_index}, "
                    f"{end_index}) and modules {module_partition_names[module_partition]}."
                )

                max_total_examples = total_data_examples // factor_args.covariance_data_partition_size
                if max_total_examples < self.state.num_processes:
                    error_msg = "The number of processes are more than the data examples."
                    self.logger.error(error_msg)
                    raise ValueError(error_msg)
                if per_device_batch_size is None:
                    per_device_batch_size = self._find_executable_covariance_factors_batch_size(
                        dataloader_params=dataloader_params,
                        dataset=dataset,
                        factor_args=factor_args,
                        total_data_examples=max_total_examples,
                        tracked_module_names=module_partition_names[0],
                    )
                covariance_factors = self._fit_partitioned_covariance_matrices(
                    dataset=dataset,
                    per_device_batch_size=per_device_batch_size,
                    dataloader_params=dataloader_params,
                    factor_args=factor_args,
                    indices=list(range(start_index, end_index)),
                    tracked_module_names=module_partition_names[module_partition],
                )
                with self.profiler.profile("Save Covariance"):
                    if self.state.is_main_process:
                        save_covariance_matrices(
                            output_dir=factors_output_dir,
                            covariance_factors=covariance_factors,
                            partition=(data_partition, module_partition),
                        )
                    self.state.wait_for_everyone()
                del covariance_factors
                self.logger.info(f"Saved partitioned covariance matrices at {factors_output_dir}.")

        all_end_time = get_time(state=self.state)
        elapsed_time = all_end_time - all_start_time
        self.logger.info(f"Fitted all partitioned covariance matrices in {elapsed_time:.2f} seconds.")
        aggregated_covariance_factors = self.aggregate_covariance_matrices(
            factors_name=factors_name, factor_args=factor_args
        )

        profile_summary = self.profiler.summary()
        if profile_summary != "":
            self.logger.info(self.profiler.summary())
        return aggregated_covariance_factors

    @torch.no_grad()
    def aggregate_covariance_matrices(
        self,
        factors_name: str,
        factor_args: FactorArguments,
    ) -> Optional[FACTOR_TYPE]:
        """Aggregates covariance matrices computed for all data and module partitions."""
        factors_output_dir = self.factors_output_dir(factors_name=factors_name)
        if not factors_output_dir.exists():
            error_msg = (
                f"Factors output directory {factors_output_dir} is not found "
                f"when trying to aggregate partitioned covariance matrices."
            )
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        data_partition_size = factor_args.covariance_data_partition_size
        module_partition_size = factor_args.covariance_module_partition_size
        all_required_partitions = [(i, j) for i in range(data_partition_size) for j in range(module_partition_size)]
        all_partition_exists = all(
            covariance_matrices_exist(output_dir=factors_output_dir, partition=partition)
            for partition in all_required_partitions
        )
        if not all_partition_exists:
            self.logger.info(
                "Covariance matrices are not aggregated as covariance matrices for some partitions "
                "are not yet computed."
            )
            return

        start_time = get_time(state=self.state)
        with self.profiler.profile("Aggregate Covariance"):
            if self.state.is_main_process:
                aggregated_covariance_factors: FACTOR_TYPE = {}
                for data_partition in range(data_partition_size):
                    for module_partition in range(module_partition_size):
                        loaded_covariance_factors = load_covariance_matrices(
                            output_dir=factors_output_dir,
                            partition=(data_partition, module_partition),
                        )
                        aggregated_covariance_factors = self._aggregate_factors(
                            aggregated_factors=aggregated_covariance_factors,
                            loaded_factors=loaded_covariance_factors,
                        )
                        del loaded_covariance_factors
                with self.profiler.profile("Save Covariance"):
                    save_covariance_matrices(
                        output_dir=factors_output_dir,
                        covariance_factors=aggregated_covariance_factors,
                    )
            self.state.wait_for_everyone()
        end_time = get_time(state=self.state)
        elapsed_time = end_time - start_time
        self.logger.info(f"Aggregated all partitioned covariance matrices in {elapsed_time:.2f} seconds.")
        return aggregated_covariance_factors
