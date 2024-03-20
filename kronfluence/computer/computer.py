import logging
import os
from abc import ABC
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn import DataParallel
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.utils import data
from torch.utils.data import DistributedSampler, SequentialSampler

from kronfluence.arguments import Arguments, FactorArguments, ScoreArguments
from kronfluence.factor.covariance import (
    covariance_matrices_exist,
    load_covariance_matrices,
)
from kronfluence.factor.eigen import (
    eigendecomposition_exist,
    lambda_matrices_exist,
    load_eigendecomposition,
    load_lambda_matrices,
)
from kronfluence.module.utils import get_tracked_module_names, make_modules_partition
from kronfluence.score.pairwise import load_pairwise_scores, pairwise_scores_exist
from kronfluence.score.self import load_self_scores, self_scores_exist
from kronfluence.task import Task
from kronfluence.utils.constants import FACTOR_TYPE, SCORE_TYPE
from kronfluence.utils.dataset import (
    DataLoaderKwargs,
    DistributedEvalSampler,
    DistributedSamplerWithStack,
    make_indices_partition,
)
from kronfluence.utils.exceptions import (
    FactorsNotFoundError,
    TrackedModuleNotFoundError,
)
from kronfluence.utils.logger import PassThroughProfiler, Profiler, get_logger
from kronfluence.utils.save import (
    FACTOR_ARGUMENTS_NAME,
    FACTOR_SAVE_PREFIX,
    SCORE_ARGUMENTS_NAME,
    SCORE_SAVE_PREFIX,
    load_json,
    save_json,
)
from kronfluence.utils.state import State


class Computer(ABC):
    """A base class for computer, which computes various quantities for the given model and task."""

    def __init__(
        self,
        name: str,
        model: nn.Module,
        task: Task,
        output_dir: str,
        cpu: bool = False,
        log_level: Optional[int] = logging.INFO,
        log_main_process_only: bool = True,
        profile: bool = False,
    ) -> None:
        """Initializes an instance of the Computer class."""
        self.state = State(cpu=cpu)

        # Create and configure logger.
        disable_log = log_main_process_only and self.state.process_index != 0
        self.logger = get_logger(name=__name__, log_level=log_level, disable_log=disable_log)

        self.model = model
        self.task = task

        tracked_module_names = get_tracked_module_names(self.model)
        if len(tracked_module_names) == 0:
            error_msg = (
                f"No tracked modules found in the provided model: {self.model}. "
                f"Please make sure to run `prepare_model` before passing it in to the "
                f"Analyzer."
            )
            self.logger.error(error_msg)
            raise TrackedModuleNotFoundError(error_msg)
        self.logger.info(f"Tracking modules with names: {tracked_module_names}.")

        if self.state.use_distributed and not isinstance(model, (DDP, FSDP)):
            warning_msg = (
                "Creating a DDP module. If specific configuration needs to be used "
                "for DDP, please pass in the model after the manual DDP wrapping."
            )
            self.logger.warning(warning_msg)
            self.model.to(self.state.device)
            self.model = DDP(
                self.model,
                device_ids=[self.state.local_process_index],
                output_device=self.state.local_process_index,
            )

        if cpu and isinstance(model, (DataParallel, DDP, FSDP)):
            error_msg = "To enforce CPU, the model should not be wrapped with DP, DDP, or FSDP."
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        if not self.state.use_distributed:
            self.model.to(self.state.device)

        # Create and configure profiler.
        self.profiler = Profiler(state=self.state) if profile else PassThroughProfiler(state=self.state)

        # Create and configure output directory.
        self.output_dir = Path(output_dir).joinpath(name).resolve()
        os.makedirs(name=self.output_dir, exist_ok=True)

        # DataLoader parameters.
        self._dataloader_params = DataLoaderKwargs()

    def factors_output_dir(self, factors_name: str) -> Path:
        """Generates an output directory for storing all factors."""
        return (self.output_dir / (FACTOR_SAVE_PREFIX + factors_name)).resolve()

    def scores_output_dir(self, scores_name: str) -> Path:
        """Generates an output directory for storing all scores."""
        return (self.output_dir / (SCORE_SAVE_PREFIX + scores_name)).resolve()

    def _save_arguments(
        self,
        arguments_name: str,
        arguments: Arguments,
        output_dir: Path,
        overwrite_output_dir: bool = False,
    ) -> None:
        """Saves arguments at the specified path."""
        arguments_save_path = output_dir / f"{arguments_name}_arguments.json"
        if arguments_save_path.exists() and not overwrite_output_dir:
            self.logger.info(f"Found existing saved arguments at `{arguments_save_path}`.")
            loaded_arguments = load_json(arguments_save_path)
            if loaded_arguments != arguments.to_dict():
                error_msg = (
                    "Attempting to use the arguments that differs from the one already saved. "
                    "Please set `overwrite_output_dir=True` to overwrite existing experiment."
                )
                error_msg += f"\nNew arguments: {arguments.to_dict()}."
                error_msg += f"\nSaved arguments: {loaded_arguments}."
                self.logger.error(error_msg)
                raise ValueError(error_msg)
        else:
            save_json(arguments.to_dict(), arguments_save_path)
            self.logger.info(f"Saved arguments at `{arguments_save_path}`.")

    def _save_dataset_metadata(
        self,
        dataset_name: str,
        dataset: data.Dataset,
        output_dir: Path,
        indices: Optional[Sequence[int]] = None,
        overwrite_output_dir: bool = False,
    ) -> None:
        """Saves dataset metadata at the specified path."""
        dataset_metadata_save_path = output_dir / f"{dataset_name}_dataset_metadata.json"
        dataset_metadata = {
            "type": type(dataset).__name__,
            "dataset_size": len(dataset),
            "indices": indices,
        }

        if dataset_metadata_save_path.exists() and not overwrite_output_dir:
            self.logger.info(f"Found existing saved dataset metadata at `{dataset_metadata_save_path}`.")
            # Load the existing dataset metadata for comparison.
            loaded_metadata = load_json(dataset_metadata_save_path)
            if loaded_metadata != dataset_metadata:
                error_msg = (
                    "Attempting to use the dataset that differs from the one already saved. "
                    "Please set `overwrite_output_dir=True` to overwrite existing experiment."
                )
                error_msg += f"\nNew metadata: {dataset_metadata}."
                error_msg += f"\nSaved metadata: {loaded_metadata}."
                self.logger.error(error_msg)
                raise ValueError(error_msg)
        else:
            save_json(dataset_metadata, dataset_metadata_save_path)
            self.logger.info(f"Saved dataset metadata at `{dataset_metadata_save_path}`.")

    def _get_dataloader(
        self,
        dataset: data.Dataset,
        per_device_batch_size: int,
        dataloader_params: Dict[str, Any],
        indices: Optional[Sequence[int]] = None,
        allow_duplicates: bool = False,
        stack: bool = False,
    ) -> data.DataLoader:
        """Returns the DataLoader for the given dataset, per_device_batch_size, and additional parameters."""
        if indices is not None:
            dataset = data.Subset(dataset=dataset, indices=indices)

        if self.state.use_distributed and not allow_duplicates:
            if stack:
                error_msg = "DistributedEvalSampler is not currently supported with `stack=True`."
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            sampler = DistributedEvalSampler(
                dataset=dataset,
                num_replicas=self.state.num_processes,
                rank=self.state.process_index,
            )
        elif self.state.use_distributed and allow_duplicates and stack:
            sampler = DistributedSamplerWithStack(
                dataset=dataset,
                num_replicas=self.state.num_processes,
                rank=self.state.process_index,
            )
        elif self.state.use_distributed and allow_duplicates:
            sampler = DistributedSampler(
                dataset=dataset,
                num_replicas=self.state.num_processes,
                rank=self.state.process_index,
                shuffle=False,
                drop_last=False,
            )
        else:
            sampler = SequentialSampler(data_source=dataset)

        self.logger.debug(f"Using sampler {sampler} for the DataLoader.")
        dataloader_params = {
            "batch_size": per_device_batch_size,
            "sampler": sampler,
            "drop_last": False,
        } | dataloader_params
        return data.DataLoader(dataset=dataset, **dataloader_params)

    def _configure_dataloader(self, dataloader_kwargs: DataLoaderKwargs) -> Dict[str, Any]:
        """Configures the DataLoader, logging appropriate messages."""
        if dataloader_kwargs is None:
            dataloader_kwargs = self._dataloader_params
            self.logger.info(f"DataLoader arguments not provided. Using the configuration: {dataloader_kwargs}.")
        else:
            self.logger.info(f"Using the DataLoader parameters: {dataloader_kwargs.to_dict()}.")
        return dataloader_kwargs.to_dict()

    def _get_data_partition(
        self,
        total_data_examples: int,
        data_partition_size: int,
        target_data_partitions: Optional[Union[int, List[int]]],
    ) -> Tuple[List[Tuple[int, int]], List[int]]:
        """Partitions the dataset into several chunks."""
        if total_data_examples < data_partition_size:
            error_msg = (
                f"Data partition size ({data_partition_size}) cannot be greater than the "
                f"total data points ({total_data_examples}). Please reduce the data partition "
                f"size in the argument."
            )
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        indices_partitions = make_indices_partition(
            total_data_examples=total_data_examples, partition_size=data_partition_size
        )

        if target_data_partitions is None:
            target_data_partitions = list(range(data_partition_size))

        if isinstance(target_data_partitions, int):
            target_data_partitions = [target_data_partitions]

        for data_partition in target_data_partitions:
            if data_partition < 0 or data_partition > data_partition_size:
                error_msg = (
                    f"Invalid data partition {data_partition} encountered. "
                    f"The module partition needs to be in between [0, {data_partition_size})."
                )
                self.logger.error(error_msg)
                raise ValueError(error_msg)

        return indices_partitions, target_data_partitions

    def _get_module_partition(
        self,
        module_partition_size: int,
        target_module_partitions: Optional[Union[int, List[int]]],
    ) -> Tuple[List[List[str]], List[int]]:
        """Partitions the modules into several chunks."""
        tracked_module_names = get_tracked_module_names(self.model)

        if len(tracked_module_names) < module_partition_size:
            error_msg = (
                f"Module partition size ({module_partition_size}) cannot be greater than the "
                f"total tracked modules ({len(tracked_module_names)}). Please reduce the module partition "
                f"size in the argument."
            )
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        modules_partition_list = make_modules_partition(
            total_module_names=tracked_module_names,
            partition_size=module_partition_size,
        )

        if target_module_partitions is None:
            target_module_partitions = list(range(module_partition_size))

        if isinstance(target_module_partitions, int):
            target_module_partitions = [target_module_partitions]

        for module_partition in target_module_partitions:
            if module_partition < 0 or module_partition > module_partition_size:
                error_msg = (
                    f"Invalid module partition {module_partition} encountered. "
                    f"The module partition needs to be in between [0, {module_partition_size})."
                )
                self.logger.error(error_msg)
                raise ValueError(error_msg)

        return modules_partition_list, target_module_partitions

    def _log_profile_summary(self) -> None:
        """Log the summary of the profiling results."""
        profile_summary = self.profiler.summary()
        if profile_summary != "":
            self.logger.info(self.profiler.summary())

    def load_factor_args(self, factors_name: str) -> Optional[FactorArguments]:
        """Loads factor arguments with the given factor name."""
        factors_output_dir = self.factors_output_dir(factors_name=factors_name)
        arguments_save_path = factors_output_dir / f"{FACTOR_ARGUMENTS_NAME}_arguments.json"
        if not arguments_save_path.exists():
            return None
        return FactorArguments(**load_json(arguments_save_path))

    def load_covariance_matrices(self, factors_name: str) -> Optional[FACTOR_TYPE]:
        """Loads covariance matrices with the given factor name."""
        factors_output_dir = self.factors_output_dir(factors_name=factors_name)
        if not covariance_matrices_exist(output_dir=factors_output_dir):
            return None
        return load_covariance_matrices(output_dir=factors_output_dir)

    def load_eigendecomposition(self, factors_name: str) -> Optional[FACTOR_TYPE]:
        """Loads Eigendecomposition results with the given factor name."""
        factors_output_dir = self.factors_output_dir(factors_name=factors_name)
        if not eigendecomposition_exist(output_dir=factors_output_dir):
            return None
        return load_eigendecomposition(output_dir=factors_output_dir)

    def load_lambda_matrices(self, factors_name: str) -> Optional[FACTOR_TYPE]:
        """Loads Lambda matrices with the given factor name."""
        factors_output_dir = self.factors_output_dir(factors_name=factors_name)
        if not lambda_matrices_exist(output_dir=factors_output_dir):
            return None
        return load_lambda_matrices(output_dir=factors_output_dir)

    def load_score_args(self, scores_name: str) -> Optional[ScoreArguments]:
        """Loads score arguments with the given score name."""
        scores_output_dir = self.scores_output_dir(scores_name=scores_name)
        arguments_save_path = scores_output_dir / f"{SCORE_ARGUMENTS_NAME}_arguments.json"
        if not arguments_save_path.exists():
            return None
        return ScoreArguments(**load_json(arguments_save_path))

    def load_pairwise_scores(self, scores_name: str) -> Optional[SCORE_TYPE]:
        """Loads pairwise scores with the given score name."""
        scores_output_dir = self.scores_output_dir(scores_name=scores_name)
        if pairwise_scores_exist(output_dir=scores_output_dir):
            return load_pairwise_scores(output_dir=scores_output_dir)
        return None

    def load_self_scores(self, scores_name: str) -> Optional[SCORE_TYPE]:
        """Loads self-influence scores with the given score name."""
        scores_output_dir = self.scores_output_dir(scores_name=scores_name)
        if self_scores_exist(output_dir=scores_output_dir):
            return load_self_scores(output_dir=scores_output_dir)
        return None

    def load_all_factors(self, factors_name: str) -> FACTOR_TYPE:
        """Loads factors from disk."""
        from kronfluence.factor.config import (  # pylint: disable=import-outside-toplevel
            FactorConfig,
        )

        factor_args = self.load_factor_args(factors_name)
        factors_output_dir = self.factors_output_dir(factors_name=factors_name)
        if factor_args is None:
            error_msg = f"Factors with name `{factors_name}` was not found at `{factors_output_dir}`."
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        loaded_factors: FACTOR_TYPE = {}
        factor_config = FactorConfig.CONFIGS[factor_args.strategy]
        if factor_config.requires_covariance_matrices_for_precondition:
            covariance_factors = self.load_covariance_matrices(factors_name=factors_name)
            if covariance_factors is None:
                error_msg = (
                    f"Strategy `{factor_args.strategy}` computing covariance matrices. "
                    f"However, the covariance matrices were not found."
                )
                self.logger.error(error_msg)
                raise FactorsNotFoundError(error_msg)
            loaded_factors.update(covariance_factors)

        if factor_config.requires_eigendecomposition_for_precondition:
            eigen_factors = self.load_eigendecomposition(factors_name=factors_name)
            if eigen_factors is None:
                error_msg = (
                    f"Strategy `{factor_args.strategy}` computing Eigendecomposition. "
                    f"However, the Eigendecomposition results were not found."
                )
                self.logger.error(error_msg)
                raise FactorsNotFoundError(error_msg)
            loaded_factors.update(eigen_factors)

        if factor_config.requires_lambda_matrices_for_precondition:
            lambda_factors = self.load_lambda_matrices(factors_name=factors_name)
            if lambda_factors is None:
                error_msg = (
                    f"Strategy `{factor_args.strategy}` computing Lambda matrices. "
                    f"However, the Lambda matrices were not found."
                )
                self.logger.error(error_msg)
                raise FactorsNotFoundError(error_msg)
            loaded_factors.update(lambda_factors)
        return loaded_factors
