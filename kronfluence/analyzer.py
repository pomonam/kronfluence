import copy
from pathlib import Path
from typing import Dict, Optional, Union

import torch
from accelerate.utils import extract_model_from_parallel
from safetensors.torch import save_file
from torch import nn
from torch.utils import data

from kronfluence.arguments import FactorArguments
from kronfluence.computer.factor_computer import FactorComputer
from kronfluence.computer.score_computer import ScoreComputer
from kronfluence.module.utils import wrap_tracked_modules
from kronfluence.task import Task
from kronfluence.utils.dataset import DataLoaderKwargs
from kronfluence.utils.save import load_file, verify_models_equivalence


def prepare_model(
    model: nn.Module,
    task: Task,
) -> nn.Module:
    """Prepares the model for analysis by setting all parameters and buffers to non-trainable
    and installing `TrackedModule` wrappers on supported modules.

    Args:
        model (nn.Module):
            The PyTorch model to be prepared for analysis.
        task (Task):
            The specific task associated with the model, used for `TrackedModule` installation.

    Returns:
        nn.Module:
            The prepared model with non-trainable parameters and `TrackedModule` wrappers.
    """
    model.eval()
    for params in model.parameters():
        params.requires_grad = False
    for buffers in model.buffers():
        buffers.requires_grad = False

    # Install `TrackedModule` wrappers on supported modules.
    model = wrap_tracked_modules(model=model, task=task)
    return model


class Analyzer(FactorComputer, ScoreComputer):
    """Handles the computation of factors (e.g., covariance matrices) and scores for a given PyTorch model."""

    def __init__(
        self,
        analysis_name: str,
        model: nn.Module,
        task: Task,
        cpu: bool = False,
        log_level: Optional[int] = None,
        log_main_process_only: bool = True,
        profile: bool = False,
        disable_tqdm: bool = False,
        output_dir: str = "./influence_results",
        disable_model_save: bool = True,
    ) -> None:
        """Initializes an instance of the `Analyzer` class.

        Args:
            analysis_name (str):
                Unique identifier for the analysis, used for organizing results.
            model (nn.Module):
                The PyTorch model to be analyzed.
            task (Task):
                The specific task associated with the model.
            cpu (bool, optional):
                If `True`, forces analysis to be performed on CPU. Defaults to `False`.
            log_level (int, optional):
                Logging level (e.g., logging.DEBUG, logging.INFO). Defaults to root logging level.
            log_main_process_only (bool, optional):
                If `True`, restricts logging to the main process. Defaults to `True`.
            profile (bool, optional):
                If `True`, enables performance profiling logs. Defaults to `False`.
            disable_tqdm (bool, optional):
                If `True`, disables TQDM progress bars. Defaults to `False`.
            output_dir (str):
                Directory path for storing analysis results. Defaults to './influence_results'.
            disable_model_save (bool, optional):
                If `True`, prevents saving the model's `state_dict`. Defaults to `True`.

        Raises:
            ValueError:
                If the provided model differs from a previously saved model when `disable_model_save=False`.
        """
        super().__init__(
            name=analysis_name,
            model=model,
            task=task,
            cpu=cpu,
            log_level=log_level,
            log_main_process_only=log_main_process_only,
            profile=profile,
            disable_tqdm=disable_tqdm,
            output_dir=output_dir,
        )
        self.logger.info(f"Initializing `Analyzer` with parameters: {locals()}")
        self.logger.info(f"Process state configuration:\n{repr(self.state)}")

        # Save model parameters if necessary.
        if self.state.is_main_process and not disable_model_save:
            self._save_model()
        self.state.wait_for_everyone()

    def set_dataloader_kwargs(self, dataloader_kwargs: DataLoaderKwargs) -> None:
        """Sets the default DataLoader arguments.

        Args:
            dataloader_kwargs (DataLoaderKwargs):
                The object containing arguments for PyTorch DataLoader.
        """
        self._dataloader_params = dataloader_kwargs

    @torch.no_grad()
    def _save_model(self) -> None:
        """Saves the model to the output directory."""
        model_save_path = self.output_dir / "model.safetensors"
        extracted_model = extract_model_from_parallel(model=copy.deepcopy(self.model), keep_fp32_wrapper=True)

        if model_save_path.exists():
            self.logger.info(f"Found existing saved model at `{model_save_path}`.")
            # Load existing model's `state_dict` for comparison.
            loaded_state_dict = load_file(model_save_path)
            if not verify_models_equivalence(loaded_state_dict, extracted_model.state_dict()):
                error_msg = (
                    "Detected a difference between the current model and the one saved at "
                    f"`{model_save_path}`. Consider using a different `analysis_name` to avoid conflicts."
                )
                self.logger.error(error_msg)
                raise ValueError(error_msg)
        else:
            self.logger.info(f"No existing model found at `{model_save_path}`.")
            state_dict = extracted_model.state_dict()
            state_dict = {k: v.clone().contiguous() for k, v in state_dict.items()}
            save_file(state_dict, model_save_path)
            self.logger.info(f"Saved model at `{model_save_path}`.")

    def fit_all_factors(
        self,
        factors_name: str,
        dataset: data.Dataset,
        per_device_batch_size: Optional[int] = None,
        initial_per_device_batch_size_attempt: int = 4096,
        dataloader_kwargs: Optional[DataLoaderKwargs] = None,
        factor_args: Optional[FactorArguments] = None,
        overwrite_output_dir: bool = False,
    ) -> None:
        """Computes all necessary factors for the given strategy.

        Args:
            factors_name (str):
                Unique identifier for the factor, used for organizing results.
            dataset (data.Dataset):
                Dataset used to fit all influence factors.
            per_device_batch_size (int, optional):
                Per-device batch size for factor fitting. If not specified, executable per-device batch size
                is automatically determined.
            initial_per_device_batch_size_attempt (int):
                Initial batch size attempt when `per_device_batch_size` is not explicitly provided. Defaults to `4096`.
            dataloader_kwargs (DataLoaderKwargs, optional):
                Additional arguments for PyTorch's DataLoader.
            factor_args (FactorArguments, optional):
                Arguments for factor computation. Defaults to `FactorArguments` default values.
            overwrite_output_dir (bool, optional):
                If `True`, overwrites existing factors with the same `factors_name`. Defaults to `False`.
        """
        self.fit_covariance_matrices(
            factors_name=factors_name,
            dataset=dataset,
            per_device_batch_size=per_device_batch_size,
            initial_per_device_batch_size_attempt=initial_per_device_batch_size_attempt,
            dataloader_kwargs=dataloader_kwargs,
            factor_args=factor_args,
            overwrite_output_dir=overwrite_output_dir,
        )
        self.perform_eigendecomposition(
            factors_name=factors_name,
            factor_args=factor_args,
            overwrite_output_dir=overwrite_output_dir,
        )
        self.fit_lambda_matrices(
            factors_name=factors_name,
            dataset=dataset,
            per_device_batch_size=per_device_batch_size,
            initial_per_device_batch_size_attempt=initial_per_device_batch_size_attempt,
            dataloader_kwargs=dataloader_kwargs,
            factor_args=factor_args,
            overwrite_output_dir=overwrite_output_dir,
        )

    @staticmethod
    def load_file(path: Union[str, Path]) -> Dict[str, torch.Tensor]:
        """Loads a `safetensors` file from the given path.

        Args:
            path (Path):
                The path to the `safetensors` file.

        Returns:
            Dict[str, torch.Tensor]:
                Dictionary mapping strings to tensors from the loaded file.

        Raises:
            FileNotFoundError:
                If the specified file does not exist.

        Note:
            For more information on `safetensors`, see https://github.com/huggingface/safetensors.
        """
        if isinstance(path, str):
            path = Path(path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}.")
        return load_file(path)

    @staticmethod
    def get_module_summary(model: nn.Module) -> str:
        """Generates a formatted summary of the model's modules, excluding those without parameters. This summary is
        useful for identifying which modules to compute influence scores for.

        Args:
            model (nn.Module):
                The PyTorch model to be summarized.

        Returns:
            str:
                A formatted string containing the model summary, including module names and representations.
        """
        format_str = "==Model Summary=="
        for module_name, module in model.named_modules():
            if len(list(module.children())) > 0:
                continue
            if len(list(module.parameters())) == 0:
                continue
            format_str += f"\nModule Name: `{module_name}`, Module: {repr(module)}"
        return format_str
