import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Union

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
from kronfluence.factor.eigen import (
    eigendecomposition_exist,
    fit_lambda_matrices_with_loader,
    lambda_matrices_exist,
    load_eigendecomposition,
    load_lambda_matrices,
    perform_eigendecomposition,
    save_eigendecomposition,
    save_lambda_matrices,
)
from kronfluence.module.tracked_module import ModuleMode
from kronfluence.module.utils import set_mode
from kronfluence.utils.constants import FACTOR_TYPE
from kronfluence.utils.dataset import DataLoaderKwargs, find_executable_batch_size
from kronfluence.utils.exceptions import FactorsNotFoundError
from kronfluence.utils.logger import get_time
from kronfluence.utils.save import FACTOR_ARGUMENTS_NAME
from kronfluence.utils.state import release_memory


class FactorComputer(Computer):
    """Handles the computation of all factors for a given PyTorch model."""

    def _configure_and_save_factor_args(
        self, factor_args: Optional[FactorArguments], factors_output_dir: Path, overwrite_output_dir: bool
    ) -> FactorArguments:
        """Configure the provided factor arguments and save it in disk."""
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
        self.state.wait_for_everyone()
        return factor_args

    @torch.no_grad()
    def _aggregate_factors(
        self,
        factors_name: str,
        data_partition_size: int,
        module_partition_size: int,
        exists_fnc: Callable,
        load_fnc: Callable,
        save_fnc: Callable,
    ) -> Optional[FACTOR_TYPE]:
        """Aggregates factors computed for all data and module partitions."""
        factors_output_dir = self.factors_output_dir(factors_name=factors_name)
        if not factors_output_dir.exists():
            error_msg = f"Factors directory `{factors_output_dir}` is not found when trying to aggregate factors."
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        all_required_partitions = [(i, j) for i in range(data_partition_size) for j in range(module_partition_size)]
        all_partition_exists = all(
            exists_fnc(output_dir=factors_output_dir, partition=partition) for partition in all_required_partitions
        )
        if not all_partition_exists:
            self.logger.warning("Factors are not aggregated as factors for some partitions are not yet computed.")
            return

        start_time = time.time()
        aggregated_factors: FACTOR_TYPE = {}
        for data_partition in range(data_partition_size):
            for module_partition in range(module_partition_size):
                loaded_factors = load_fnc(
                    output_dir=factors_output_dir,
                    partition=(data_partition, module_partition),
                )
                for factor_name, factors in loaded_factors.items():
                    if factor_name not in aggregated_factors:
                        aggregated_factors[factor_name]: Dict[str, torch.Tensor] = {}

                    for module_name in factors:
                        if module_name not in aggregated_factors[factor_name]:
                            aggregated_factors[factor_name][module_name] = factors[module_name]
                        else:
                            aggregated_factors[factor_name][module_name].add_(factors[module_name])
                del loaded_factors
        save_fnc(
            output_dir=factors_output_dir,
            factors=aggregated_factors,
        )
        end_time = time.time()
        elapsed_time = end_time - start_time
        self.logger.info(f"Aggregated all factors in {elapsed_time:.2f} seconds.")

    def _find_executable_factors_batch_size(
        self,
        func: Callable,
        func_kwargs: Dict[str, Any],
        initial_per_device_batch_size_attempt: int,
        dataset: data.Dataset,
        dataloader_params: Dict[str, Any],
        total_data_examples: Optional[int] = None,
    ) -> int:
        """Automatically finds executable batch size for performing `func`."""
        if self.state.use_distributed:
            error_msg = (
                "Automatic batch size search is currently not supported for multi-GPU training. "
                "Please manually configure the batch size by passing in `per_device_batch_size`."
            )
            self.logger.error(error_msg)
            raise NotImplementedError(error_msg)

        self.logger.info("Automatically determining executable batch size.")
        if total_data_examples is None:
            total_data_examples = len(dataset)
        start_batch_size = min(
            [
                initial_per_device_batch_size_attempt,
                total_data_examples,
            ]
        )

        def executable_batch_size_func(batch_size: int) -> None:
            self.logger.info(f"Attempting to set per-device batch size to {batch_size}.")
            # Release all memory that could be caused by the previous OOM.
            set_mode(model=self.model, mode=ModuleMode.DEFAULT, keep_factors=False)
            self.model.zero_grad(set_to_none=True)
            release_memory()
            total_batch_size = batch_size * self.state.num_processes
            loader = self._get_dataloader(
                dataset=dataset,
                per_device_batch_size=batch_size,
                indices=list(range(total_batch_size)),
                dataloader_params=dataloader_params,
                allow_duplicates=True,
            )
            func(loader=loader, **func_kwargs)

        per_device_batch_size = find_executable_batch_size(
            func=executable_batch_size_func,
            start_batch_size=start_batch_size,
        )
        self.logger.info(f"Executable batch size determined: {per_device_batch_size}.")
        return per_device_batch_size

    def fit_covariance_matrices(
        self,
        factors_name: str,
        dataset: data.Dataset,
        per_device_batch_size: Optional[int] = None,
        initial_per_device_batch_size_attempt: int = 4096,
        dataloader_kwargs: Optional[DataLoaderKwargs] = None,
        factor_args: Optional[FactorArguments] = None,
        target_data_partitions: Optional[Union[Sequence[int], int]] = None,
        target_module_partitions: Optional[Union[Sequence[int], int]] = None,
        overwrite_output_dir: bool = False,
    ) -> None:
        """Computes activation and pseudo-covariance matrices with the given dataset.

        Args:
            factors_name (str):
                The unique identifier for the factor, used to organize and retrieve the results.
            dataset (data.Dataset):
                The dataset that will be used to fit covariance matrices.
            per_device_batch_size (int, optional):
                The per-device batch size used to fit the factors. If not specified, executable
                batch size is automatically determined.
            initial_per_device_batch_size_attempt (int, optional):
                The initial attempted per-device batch size when the batch size is not provided.
            dataloader_kwargs (DataLoaderKwargs, optional):
                Controls additional arguments for PyTorch's DataLoader.
            factor_args (FactorArguments, optional):
                Arguments related to computing the factors. If not specified, the default values of
                `FactorArguments` will be used.
            target_data_partitions(Sequence[int], int, optional):
                The list of data partition to fit covariance matrices. By default, covariance
                matrices will be computed for all partitions.
            target_module_partitions(Sequence[int], int, optional):
                The list of module partition to fit covariance matrices. By default, covariance
                matrices will be computed for all partitions.
            overwrite_output_dir (bool, optional):
                If True, the existing factors with the same `factors_name` will be overwritten.
        """
        self.logger.debug(f"Fitting covariance matrices with parameters: {locals()}")

        factors_output_dir = self.factors_output_dir(factors_name=factors_name)
        os.makedirs(factors_output_dir, exist_ok=True)
        if covariance_matrices_exist(output_dir=factors_output_dir) and not overwrite_output_dir:
            self.logger.info(f"Found existing covariance matrices at `{factors_output_dir}`. Skipping.")
            return

        factor_args = self._configure_and_save_factor_args(
            factor_args=factor_args, factors_output_dir=factors_output_dir, overwrite_output_dir=overwrite_output_dir
        )

        if not FactorConfig.CONFIGS[factor_args.strategy].requires_covariance_matrices:
            self.logger.info(
                f"Strategy `{factor_args.strategy}` does not require fitting covariance matrices. Skipping."
            )
            return

        dataloader_params = self._configure_dataloader(dataloader_kwargs)
        if self.state.is_main_process:
            self._save_dataset_metadata(
                dataset_name="covariance",
                dataset=dataset,
                output_dir=factors_output_dir,
                overwrite_output_dir=overwrite_output_dir,
            )

        if factor_args.covariance_max_examples is None:
            total_data_examples = len(dataset)
        else:
            total_data_examples = min([factor_args.covariance_max_examples, len(dataset)])
        self.logger.info(f"Total data examples to fit covariance matrices: {total_data_examples}.")

        no_partition = (
            factor_args.covariance_data_partition_size == 1 and factor_args.covariance_module_partition_size == 1
        )
        partition_provided = target_data_partitions is not None or target_module_partitions is not None
        if no_partition and partition_provided:
            error_msg = (
                "`target_data_partitions` or `target_module_partitions` were specified, while"
                "the `FactorArguments` did not expect any data and module partition to compute covariance matrices."
            )
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        data_partition_indices, target_data_partitions = self._get_data_partition(
            total_data_examples=total_data_examples,
            data_partition_size=factor_args.covariance_data_partition_size,
            target_data_partitions=target_data_partitions,
        )
        max_partition_examples = total_data_examples // factor_args.covariance_data_partition_size
        module_partition_names, target_module_partitions = self._get_module_partition(
            module_partition_size=factor_args.covariance_module_partition_size,
            target_module_partitions=target_module_partitions,
        )

        if max_partition_examples < self.state.num_processes:
            error_msg = "The number of processes are more than the data examples. Try reducing the number of processes."
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        all_start_time = get_time(state=self.state)
        for data_partition in target_data_partitions:
            for module_partition in target_module_partitions:
                if no_partition:
                    partition = None
                else:
                    partition = (data_partition, module_partition)

                if (
                    covariance_matrices_exist(
                        output_dir=factors_output_dir,
                        partition=partition,
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
                    f"Fitting covariance matrices with data indices ({start_index}, {end_index}) and "
                    f"modules {module_partition_names[module_partition]}."
                )

                if per_device_batch_size is None:
                    kwargs = {
                        "model": self.model,
                        "state": self.state,
                        "task": self.task,
                        "factor_args": factor_args,
                        "tracked_module_names": module_partition_names[module_partition],
                    }
                    per_device_batch_size = self._find_executable_factors_batch_size(
                        func=fit_covariance_matrices_with_loader,
                        func_kwargs=kwargs,
                        dataset=dataset,
                        initial_per_device_batch_size_attempt=initial_per_device_batch_size_attempt,
                        dataloader_params=dataloader_params,
                        total_data_examples=max_partition_examples,
                    )

                release_memory()
                start_time = get_time(state=self.state)
                with self.profiler.profile("Fit Covariance"):
                    loader = self._get_dataloader(
                        dataset=dataset,
                        per_device_batch_size=per_device_batch_size,
                        dataloader_params=dataloader_params,
                        indices=list(range(start_index, end_index)),
                        allow_duplicates=False,
                    )
                    num_data_processed, covariance_factors = fit_covariance_matrices_with_loader(
                        model=self.model,
                        state=self.state,
                        task=self.task,
                        loader=loader,
                        factor_args=factor_args,
                        tracked_module_names=module_partition_names[module_partition],
                    )
                end_time = get_time(state=self.state)
                elapsed_time = end_time - start_time
                self.logger.info(
                    f"Fitted covariance matrices with {num_data_processed.item()} data points in "
                    f"{elapsed_time:.2f} seconds."
                )

                with self.profiler.profile("Save Covariance"):
                    if self.state.is_main_process:
                        save_covariance_matrices(
                            output_dir=factors_output_dir,
                            factors=covariance_factors,
                            partition=partition,
                            metadata=factor_args.to_str_dict(),
                        )
                    self.state.wait_for_everyone()
                del covariance_factors, loader
                self.logger.info(f"Saved covariance matrices at `{factors_output_dir}`.")

        all_end_time = get_time(state=self.state)
        elapsed_time = all_end_time - all_start_time
        if not no_partition:
            self.logger.info(f"Fitted all partitioned covariance matrices in {elapsed_time:.2f} seconds.")
            if self.state.is_main_process:
                self.aggregate_covariance_matrices(factors_name=factors_name)
                self.logger.info(f"Saved aggregated covariance matrices at `{factors_output_dir}`.")
            self.state.wait_for_everyone()
        self._log_profile_summary()

    @torch.no_grad()
    def aggregate_covariance_matrices(
        self,
        factors_name: str,
    ) -> None:
        """Aggregates all partitioned covariance matrices. The factors will not be aggregated if covariance matrices
        for some data or module partitions are missing.

        Args:
            factors_name (str):
                The unique identifier for the factor, used to organize and retrieve the results.
        """
        factor_args = self.load_factor_args(factors_name=factors_name)
        if factor_args is None:
            error_msg = (
                f"Arguments for factors with name `{factors_name}` was not found when trying to "
                f"aggregated covariance matrices."
            )
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        with self.profiler.profile("Aggregate Covariance"):
            self._aggregate_factors(
                factors_name=factors_name,
                data_partition_size=factor_args.covariance_data_partition_size,
                module_partition_size=factor_args.covariance_module_partition_size,
                exists_fnc=covariance_matrices_exist,
                load_fnc=load_covariance_matrices,
                save_fnc=save_covariance_matrices,
            )

    def perform_eigendecomposition(
        self,
        factors_name: str,
        factor_args: Optional[FactorArguments] = None,
        overwrite_output_dir: bool = False,
        load_from_factors_name: Optional[str] = None,
    ) -> None:
        """Performs Eigendecomposition on all covariance matrices.

        Args:
            factors_name (str):
                The unique identifier for the factor, used to organize and retrieve the results.
            factor_args (FactorArguments, optional):
                Arguments related to computing the factors. If not specified, the default values of
                `FactorArguments` will be used.
            overwrite_output_dir (bool, optional):
                If True, the existing factors with the same `factors_name` will be overwritten.
            load_from_factors_name (str, optional):
                The `factor_name` to load covariance matrices from. By default, covariance matrices with
                the same `factor_name` will be used.
        """
        self.logger.debug(f"Performing Eigendecomposition with parameters: {locals()}")

        factors_output_dir = self.factors_output_dir(factors_name=factors_name)
        os.makedirs(factors_output_dir, exist_ok=True)
        if eigendecomposition_exist(output_dir=factors_output_dir) and not overwrite_output_dir:
            self.logger.info(f"Found existing Eigendecomposition results at `{factors_output_dir}`. Skipping.")
            return

        factor_args = self._configure_and_save_factor_args(
            factor_args=factor_args, factors_output_dir=factors_output_dir, overwrite_output_dir=overwrite_output_dir
        )

        if not FactorConfig.CONFIGS[factor_args.strategy].requires_eigendecomposition:
            self.logger.info(
                f"Strategy `{factor_args.strategy}` does not require performing Eigendecomposition. Skipping."
            )
            return None

        load_factors_output_dir = factors_output_dir
        if load_from_factors_name is not None:
            self.logger.info(f"Will be loading covariance matrices from factors with name `{load_from_factors_name}`.")
            load_factors_output_dir = self.factors_output_dir(factors_name=load_from_factors_name)

        if not covariance_matrices_exist(output_dir=load_factors_output_dir):
            error_msg = (
                f"Covariance matrices not found at `{load_factors_output_dir}`. "
                f"To perform Eigendecomposition, covariance matrices need to be first computed."
            )
            self.logger.error(error_msg)
            raise FactorsNotFoundError(error_msg)

        with self.profiler.profile("Load Covariance"):
            covariance_factors = load_covariance_matrices(output_dir=load_factors_output_dir)

        if load_from_factors_name is not None and self.state.is_main_process:
            # Save the loaded covariances to the current factor output directory.
            with self.profiler.profile("Save Covariance"):
                save_covariance_matrices(output_dir=factors_output_dir, factors=covariance_factors)
            loaded_factor_args = self.load_factor_args(factors_name=load_from_factors_name)
            self._save_arguments(
                arguments_name=FACTOR_ARGUMENTS_NAME + "_loaded_covariance",
                arguments=loaded_factor_args,
                output_dir=factors_output_dir,
                overwrite_output_dir=True,
            )
        self.state.wait_for_everyone()

        eigen_factors = None
        if self.state.is_main_process:
            release_memory()
            start_time = time.time()
            with self.profiler.profile("Perform Eigendecomposition"):
                eigen_factors = perform_eigendecomposition(
                    covariance_factors=covariance_factors,
                    model=self.model,
                    state=self.state,
                    factor_args=factor_args,
                )
            end_time = time.time()
            elapsed_time = end_time - start_time
            self.logger.info(f"Performed Eigendecomposition in {elapsed_time:.2f} seconds.")

            with self.profiler.profile("Save Eigendecomposition"):
                save_eigendecomposition(
                    output_dir=factors_output_dir, factors=eigen_factors, metadata=factor_args.to_str_dict()
                )
            self.logger.info(f"Saved Eigendecomposition results at `{factors_output_dir}`.")
        self.state.wait_for_everyone()
        self._log_profile_summary()

    def fit_lambda_matrices(
        self,
        factors_name: str,
        dataset: data.Dataset,
        per_device_batch_size: Optional[int] = None,
        initial_per_device_batch_size_attempt: int = 4096,
        dataloader_kwargs: Optional[DataLoaderKwargs] = None,
        factor_args: Optional[FactorArguments] = None,
        target_data_partitions: Optional[Union[Sequence[int], int]] = None,
        target_module_partitions: Optional[Union[Sequence[int], int]] = None,
        overwrite_output_dir: bool = False,
        load_from_factors_name: Optional[str] = None,
    ) -> None:
        """Computes Lambda (corrected-eigenvalues) matrices with the given dataset.

        Args:
            factors_name (str):
                The unique identifier for the factor, used to organize and retrieve the results.
            dataset (data.Dataset):
                The dataset that will be used to fit Lambda matrices.
            per_device_batch_size (int, optional):
                The per-device batch size used to fit the factors. If not specified, executable
                batch size is automatically determined.
            initial_per_device_batch_size_attempt (int, optional):
                The initial attempted per-device batch size when the batch size is not provided.
            dataloader_kwargs (DataLoaderKwargs, optional):
                Controls additional arguments for PyTorch's DataLoader.
            factor_args (FactorArguments, optional):
                Arguments related to computing the factors. If not specified, the default values of
                `FactorArguments` will be used.
            target_data_partitions(Sequence[int], int, optional):
                The list of data partition to fit Lambda matrices. By default, Lambda
                matrices will be computed for all partitions.
            target_module_partitions(Sequence[int], int, optional):
                The list of module partition to fit Lambda matrices. By default, Lambda
                matrices will be computed for all partitions.
            overwrite_output_dir (bool, optional):
                If True, the existing factors with the same `factors_name` will be overwritten.
            load_from_factors_name (str, optional):
                The `factor_name` to load Eigendecomposition results from. By default, Eigendecomposition
                results with the same `factor_name` will be used.
        """
        self.logger.debug(f"Fitting Lambda matrices with parameters: {locals()}")

        factors_output_dir = self.factors_output_dir(factors_name=factors_name)
        os.makedirs(factors_output_dir, exist_ok=True)
        if lambda_matrices_exist(output_dir=factors_output_dir) and not overwrite_output_dir:
            self.logger.info(f"Found existing Lambda matrices at `{factors_output_dir}`. Skipping.")
            return

        factor_args = self._configure_and_save_factor_args(
            factor_args=factor_args, factors_output_dir=factors_output_dir, overwrite_output_dir=overwrite_output_dir
        )

        if not FactorConfig.CONFIGS[factor_args.strategy].requires_lambda_matrices:
            self.logger.info(f"Strategy `{factor_args.strategy}` does not require fitting Lambda matrices. Skipping.")
            return

        dataloader_params = self._configure_dataloader(dataloader_kwargs)
        if self.state.is_main_process:
            self._save_dataset_metadata(
                dataset_name="lambda",
                dataset=dataset,
                output_dir=factors_output_dir,
                overwrite_output_dir=overwrite_output_dir,
            )

        if load_from_factors_name is not None:
            self.logger.info(
                f"Will be loading Eigendecomposition results from factors with name `{load_from_factors_name}`."
            )
            load_factors_output_dir = self.factors_output_dir(factors_name=load_from_factors_name)
        else:
            load_factors_output_dir = factors_output_dir

        if (
            not eigendecomposition_exist(output_dir=load_factors_output_dir)
            and FactorConfig.CONFIGS[factor_args.strategy].requires_eigendecomposition_for_lambda
        ):
            error_msg = (
                f"Eigendecomposition results not found at `{load_factors_output_dir}`. "
                f"To fit Lambda matrices for `{factor_args.strategy}`, Eigendecomposition must be "
                f"performed before computing Lambda matrices."
            )
            self.logger.error(error_msg)
            raise FactorsNotFoundError(error_msg)

        eigen_factors = None
        if FactorConfig.CONFIGS[factor_args.strategy].requires_eigendecomposition_for_lambda:
            with self.profiler.profile("Load Eigendecomposition"):
                eigen_factors = load_eigendecomposition(output_dir=load_factors_output_dir)
            if load_from_factors_name is not None and self.state.is_main_process:
                with self.profiler.profile("Save Eigendecomposition"):
                    save_eigendecomposition(output_dir=factors_output_dir, factors=eigen_factors)
                loaded_factor_args = self.load_factor_args(factors_name=load_from_factors_name)
                self._save_arguments(
                    arguments_name=FACTOR_ARGUMENTS_NAME + "_loaded_eigendecomposition",
                    arguments=loaded_factor_args,
                    output_dir=factors_output_dir,
                    overwrite_output_dir=True,
                )
            self.state.wait_for_everyone()

        if factor_args.lambda_max_examples is None:
            total_data_examples = len(dataset)
        else:
            total_data_examples = min([factor_args.lambda_max_examples, len(dataset)])
        self.logger.info(f"Total data examples to fit Lambda matrices: {total_data_examples}.")

        no_partition = factor_args.lambda_data_partition_size == 1 and factor_args.lambda_module_partition_size == 1
        partition_provided = target_data_partitions is not None or target_module_partitions is not None
        if no_partition and partition_provided:
            error_msg = (
                "`target_data_partitions` or `target_module_partitions` were specified, while"
                "the `FactorArguments` did not expect any data and module partition to compute Lambda matrices."
            )
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        data_partition_indices, target_data_partitions = self._get_data_partition(
            total_data_examples=total_data_examples,
            data_partition_size=factor_args.lambda_data_partition_size,
            target_data_partitions=target_data_partitions,
        )
        max_partition_examples = total_data_examples // factor_args.lambda_data_partition_size
        module_partition_names, target_module_partitions = self._get_module_partition(
            module_partition_size=factor_args.lambda_module_partition_size,
            target_module_partitions=target_module_partitions,
        )

        if max_partition_examples < self.state.num_processes:
            error_msg = "The number of processes are more than the data examples. Try reducing the number of processes."
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        all_start_time = get_time(state=self.state)
        for data_partition in target_data_partitions:
            for module_partition in target_module_partitions:
                if no_partition:
                    partition = None
                else:
                    partition = (data_partition, module_partition)

                if (
                    lambda_matrices_exist(
                        output_dir=factors_output_dir,
                        partition=partition,
                    )
                    and not overwrite_output_dir
                ):
                    self.logger.info(
                        f"Found existing Lambda matrices for data partition {data_partition} "
                        f"and module partition {module_partition} at {factors_output_dir}. Skipping."
                    )
                    continue

                start_index, end_index = data_partition_indices[data_partition]
                self.logger.info(
                    f"Fitting Lambda matrices with data indices ({start_index}, {end_index}) and "
                    f"modules {module_partition_names[module_partition]}."
                )

                if per_device_batch_size is None:
                    kwargs = {
                        "eigen_factors": eigen_factors,
                        "model": self.model,
                        "state": self.state,
                        "task": self.task,
                        "factor_args": factor_args,
                        "tracked_module_names": module_partition_names[module_partition],
                    }
                    per_device_batch_size = self._find_executable_factors_batch_size(
                        func=fit_lambda_matrices_with_loader,
                        func_kwargs=kwargs,
                        dataset=dataset,
                        initial_per_device_batch_size_attempt=initial_per_device_batch_size_attempt,
                        dataloader_params=dataloader_params,
                        total_data_examples=max_partition_examples,
                    )

                release_memory()
                start_time = get_time(state=self.state)
                with self.profiler.profile("Fit Lambda"):
                    loader = self._get_dataloader(
                        dataset=dataset,
                        per_device_batch_size=per_device_batch_size,
                        dataloader_params=dataloader_params,
                        indices=list(range(start_index, end_index)),
                        allow_duplicates=False,
                    )
                    num_data_processed, lambda_factors = fit_lambda_matrices_with_loader(
                        eigen_factors=eigen_factors,
                        model=self.model,
                        state=self.state,
                        task=self.task,
                        loader=loader,
                        factor_args=factor_args,
                        tracked_module_names=module_partition_names[module_partition],
                    )
                end_time = get_time(state=self.state)
                elapsed_time = end_time - start_time
                self.logger.info(
                    f"Fitted Lambda matrices with {num_data_processed.item()} data points in "
                    f"{elapsed_time:.2f} seconds."
                )

                with self.profiler.profile("Save Lambda"):
                    if self.state.is_main_process:
                        save_lambda_matrices(
                            output_dir=factors_output_dir,
                            factors=lambda_factors,
                            partition=partition,
                            metadata=factor_args.to_str_dict(),
                        )
                    self.state.wait_for_everyone()
                del lambda_factors, loader
                self.logger.info(f"Saved Lambda matrices at `{factors_output_dir}`.")

        all_end_time = get_time(state=self.state)
        elapsed_time = all_end_time - all_start_time
        if not no_partition:
            self.logger.info(f"Fitted all partitioned Lambda matrices in {elapsed_time:.2f} seconds.")
            if self.state.is_main_process:
                self.aggregate_lambda_matrices(factors_name=factors_name)
                self.logger.info(f"Saved aggregated Lambda matrices at `{factors_output_dir}`.")
            self.state.wait_for_everyone()
        self._log_profile_summary()

    @torch.no_grad()
    def aggregate_lambda_matrices(
        self,
        factors_name: str,
    ) -> None:
        """Aggregates all partitioned Lambda matrices. The factors will not be aggregated if Lambda matrices
        for some data or module partitions are missing.

        Args:
            factors_name (str):
                The unique identifier for the factor, used to organize and retrieve the results.
        """
        factor_args = self.load_factor_args(factors_name=factors_name)
        if factor_args is None:
            error_msg = (
                f"Arguments for factors with name `{factors_name}` was not found when trying "
                f"to aggregated Lambda matrices."
            )
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        with self.profiler.profile("Aggregate Lambda"):
            self._aggregate_factors(
                factors_name=factors_name,
                data_partition_size=factor_args.lambda_data_partition_size,
                module_partition_size=factor_args.lambda_module_partition_size,
                exists_fnc=lambda_matrices_exist,
                load_fnc=load_lambda_matrices,
                save_fnc=save_lambda_matrices,
            )
