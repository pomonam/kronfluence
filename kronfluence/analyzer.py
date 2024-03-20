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
    """Prepares the model before passing it to `Analyzer`. This function sets `param.requires_grad = False`
    for all modules that does not require influence computations and installs `TrackedModule` to supported
    modules. This `TrackedModule` keeps track of relevant statistics needed to compute influence scores.

    Args:
        model (nn.Module):
            The PyTorch model to be analyzed.
        task (Task):
            The specific task associated with the model.

    Returns:
        nn.Module:
            The same PyTorch model with `param.requires_grad = False` on all modules that does not require influence
            computations and with `TrackedModule` installed.
    """
    model.eval()
    for params in model.parameters():
        params.requires_grad = False
    # Install `TrackedModule` to the model.
    model = wrap_tracked_modules(model=model, task=task)
    return model


class Analyzer(FactorComputer, ScoreComputer):
    """
    Handles the computation of all factors (e.g., covariance and Lambda matrices for EKFAC)
    and influence scores for a given PyTorch model.
    """

    def __init__(
        self,
        analysis_name: str,
        model: nn.Module,
        task: Task,
        cpu: bool = False,
        log_level: Optional[int] = None,
        log_main_process_only: bool = True,
        profile: bool = False,
        output_dir: str = "./analyses",
        disable_model_save: bool = True,
    ) -> None:
        """Initializes an instance of the Analyzer class.

        Args:
            analysis_name (str):
                The unique identifier for the analysis, used to organize and retrieve the results.
            model (nn.Module):
                The PyTorch model to be analyzed.
            task (Task):
                The specific task associated with the model.
            cpu (bool, optional):
                Specifies whether the analysis should be explicitly performed using the CPU.
                Defaults to False, utilizing GPU resources if available.
            log_level (int, optional):
                The logging level to use (e.g., logging.DEBUG, logging.INFO). Defaults to the root logging level.
            log_main_process_only (bool, optional):
                If True, restricts logging to the main process. Defaults to True.
            profile (bool, optional):
                Enables the generation of performance profiling logs. This can be useful for
                identifying bottlenecks or performance issues. Defaults to False.
            output_dir (str):
                The file path to the directory, where analysis results will be stored. If the directory
                does not exist, it will be created. Defaults to './analyses'.
            disable_model_save (bool, optional):
                If set to True, prevents the saving of the model's state_dict. When the provided model is different
                from the previously saved model, it will raise an Exception. Defaults to True.
        """
        super().__init__(
            name=analysis_name,
            model=model,
            task=task,
            cpu=cpu,
            log_level=log_level,
            log_main_process_only=log_main_process_only,
            profile=profile,
            output_dir=output_dir,
        )
        self.logger.info(f"Initializing Computer with parameters: {locals()}")
        self.logger.debug(f"Process state configuration:\n{repr(self.state)}")

        # Save model parameters.
        if self.state.is_main_process and not disable_model_save:
            self._save_model()
        self.state.wait_for_everyone()

    def set_dataloader_kwargs(self, dataloader_kwargs: DataLoaderKwargs) -> None:
        """Sets the default DataLoader parameters to use for all DataLoaders.

        Args:
            dataloader_kwargs (DataLoaderKwargs):
                The object containing parameters for DataLoader.
        """
        self._dataloader_params = dataloader_kwargs

    @torch.no_grad()
    def _save_model(self) -> None:
        """Saves the model to the output directory."""
        model_save_path = self.output_dir / "model.safetensors"
        extracted_model = extract_model_from_parallel(model=self.model, keep_fp32_wrapper=True)

        if model_save_path.exists():
            self.logger.info(f"Found existing saved model at `{model_save_path}`.")
            # Load the existing model's state_dict for comparison.
            loaded_state_dict = load_file(model_save_path)
            if not verify_models_equivalence(loaded_state_dict, extracted_model.state_dict()):
                error_msg = (
                    "Detected a difference between the current model and the one saved at "
                    f"`{model_save_path}`. Consider using a different `analysis_name` to "
                    f"avoid conflicts."
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
        """Computes all necessary factors for the given factor strategy. As an example, EK-FAC
        requires (1) computing covariance matrices, (2) performing Eigendecomposition, and
        (3) computing Lambda (corrected-eigenvalues) matrices.

        Args:
            factors_name (str):
                The unique identifier for the factor, used to organize and retrieve the results.
            dataset (data.Dataset):
                The dataset that will be used to fit all the factors.
            per_device_batch_size (int, optional):
                The per-device batch size used to fit the factors. If not specified, executable
                per-device batch size is automatically determined.
            initial_per_device_batch_size_attempt (int):
                The initial attempted per-device batch size when the batch size is not provided.
            dataloader_kwargs (DataLoaderKwargs, optional):
                Controls additional arguments for PyTorch's DataLoader.
            factor_args (FactorArguments, optional):
                Arguments related to computing the factors. If not specified,
                the default values of `FactorArguments` will be used.
            overwrite_output_dir (bool, optional):
                If True, the existing factors with the same name will be overwritten.
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
        """Loads the `.safetensors` file at the given path from disk.

        See https://github.com/huggingface/safetensors.

        Args:
            path (Path):
                The path to the `.safetensors` file.

        Returns:
            Dict[str, torch.Tensor]:
                The contents of the file, which is the dictionary mapping string to tensors.
        """
        if isinstance(path, str):
            path = Path(path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"File does not exists at `{path}`.")
        return load_file(path)

    @staticmethod
    def get_module_summary(model: nn.Module) -> str:
        """Returns the formatted summary of the modules in model. Useful identifying which modules to
        compute influence scores.

        Args:
            model (nn.Module):
                The PyTorch model to be investigated.

        Returns:
            str:
                The formatted string summary of the model.
        """
        format_str = "==Model Summary=="
        for module_name, module in model.named_modules():
            if len(list(module.children())) > 0:
                continue
            if len(list(module.parameters())) == 0:
                continue
            format_str += f"\nModule Name: `{module_name}`, Module: {repr(module)}"
        return format_str
