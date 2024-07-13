# pylint: skip-file

from typing import Callable, Dict, Optional, Tuple

import torch
from torch import nn
from torch.utils import data
from transformers import default_data_collator

from kronfluence import Analyzer, prepare_model
from kronfluence.task import Task
from kronfluence.utils.exceptions import UnsupportableModuleError
from tests.testable_tasks.classification import (
    ClassificationTask,
    WrongClassificationTask,
    make_classification_dataset,
    make_conv_bn_model,
    make_conv_inplace_model,
    make_conv_model,
)
from tests.testable_tasks.language_modeling import (
    LanguageModelingTask,
    make_gpt_dataset,
    make_tiny_gpt,
)
from tests.testable_tasks.multiple_choice import (
    MultipleChoiceTask,
    make_roberta_dataset,
    make_tiny_roberta,
)
from tests.testable_tasks.regression import (
    GradientCheckpointRegressionTask,
    RegressionTask,
    make_mlp_model,
    make_regression_dataset,
    make_repeated_mlp_model,
)
from tests.testable_tasks.text_classification import (
    TextClassificationTask,
    WrongTextClassificationTask,
    make_bert_dataset,
    make_tiny_bert,
)

RTOL = 1.3e-6
ATOL = 1e-5

DEFAULT_FACTORS_NAME = "pytest"
DEFAULT_SCORES_NAME = "pytest"


def custom_factors_name(name: str) -> str:
    return f"{DEFAULT_FACTORS_NAME}_{name}"


def custom_scores_name(name: str) -> str:
    return f"{DEFAULT_FACTORS_NAME}_{name}"


def prepare_model_and_analyzer(model: nn.Module, task: Task) -> Tuple[nn.Module, Analyzer]:
    model = prepare_model(model=model, task=task)
    analyzer = Analyzer(
        analysis_name="pytest",
        model=model,
        task=task,
        disable_model_save=True,
        disable_tqdm=True,
    )
    return model, analyzer


def prepare_test(
    test_name: str,
    train_size: int = 32,
    query_size: int = 16,
    seed: int = 0,
    do_not_pad: bool = False,
) -> Tuple[nn.Module, data.Dataset, data.Dataset, Optional[Callable], Task]:
    if test_name == "mlp":
        model = make_mlp_model(seed=seed)
        train_dataset = make_regression_dataset(num_data=train_size, seed=seed)
        query_dataset = make_regression_dataset(num_data=query_size, seed=seed + 1)
        task = RegressionTask()
        data_collator = None
    elif test_name == "repeated_mlp":
        model = make_repeated_mlp_model(seed=seed)
        train_dataset = make_regression_dataset(num_data=train_size, seed=seed)
        query_dataset = make_regression_dataset(num_data=query_size, seed=seed + 1)
        task = RegressionTask()
        data_collator = None
    elif test_name == "mlp_checkpoint":
        model = make_mlp_model(seed=seed)
        train_dataset = make_regression_dataset(num_data=train_size, seed=seed)
        query_dataset = make_regression_dataset(num_data=query_size, seed=seed + 1)
        task = GradientCheckpointRegressionTask()
        data_collator = None
    elif test_name == "conv":
        model = make_conv_model(seed=seed)
        train_dataset = make_classification_dataset(num_data=train_size, seed=seed)
        query_dataset = make_classification_dataset(num_data=query_size, seed=seed + 1)
        task = ClassificationTask()
        data_collator = None
    elif test_name == "conv_inplace":
        model = make_conv_inplace_model(seed=seed)
        train_dataset = make_classification_dataset(num_data=train_size, seed=seed)
        query_dataset = make_classification_dataset(num_data=query_size, seed=seed + 1)
        task = ClassificationTask()
        data_collator = None
    elif test_name == "wrong_conv":
        model = make_conv_model(seed=seed)
        train_dataset = make_classification_dataset(num_data=train_size, seed=seed)
        query_dataset = make_classification_dataset(num_data=query_size, seed=seed + 1)
        task = WrongClassificationTask()
        data_collator = None
    elif test_name == "conv_bn":
        model = make_conv_bn_model(seed=seed)
        train_dataset = make_classification_dataset(num_data=train_size, seed=seed)
        query_dataset = make_classification_dataset(num_data=query_size, seed=seed + 1)
        task = ClassificationTask()
        data_collator = None
    elif test_name == "bert":
        model = make_tiny_bert(seed=seed)
        train_dataset = make_bert_dataset(num_data=train_size, seed=seed, do_not_pad=do_not_pad)
        query_dataset = make_bert_dataset(num_data=query_size, seed=seed + 1, do_not_pad=do_not_pad)
        task = TextClassificationTask()
        data_collator = default_data_collator
    elif test_name == "wrong_bert":
        model = make_tiny_bert(seed=seed)
        train_dataset = make_bert_dataset(num_data=train_size, seed=seed, do_not_pad=do_not_pad)
        query_dataset = make_bert_dataset(num_data=query_size, seed=seed + 1, do_not_pad=do_not_pad)
        task = WrongTextClassificationTask()
        data_collator = default_data_collator
    elif test_name == "roberta":
        model = make_tiny_roberta(seed=seed)
        train_dataset = make_roberta_dataset(num_data=train_size, seed=seed)
        query_dataset = make_roberta_dataset(num_data=query_size, seed=seed + 1)
        task = MultipleChoiceTask()
        data_collator = default_data_collator
    elif test_name == "gpt":
        model = make_tiny_gpt(seed=seed)
        train_dataset = make_gpt_dataset(num_data=train_size, seed=seed)
        query_dataset = make_gpt_dataset(num_data=query_size, seed=seed + 1)
        task = LanguageModelingTask()
        data_collator = default_data_collator
    elif test_name == "gpt_checkpoint":
        model = make_tiny_gpt(seed=seed)
        model.gradient_checkpointing_enable()
        train_dataset = make_gpt_dataset(num_data=train_size, seed=seed)
        query_dataset = make_gpt_dataset(num_data=query_size, seed=seed + 1)
        task = LanguageModelingTask()
        data_collator = default_data_collator
    else:
        raise NotImplementedError(f"{test_name} is not a valid test configuration name.")
    model.eval()
    return model, train_dataset, query_dataset, data_collator, task


def check_tensor_dict_equivalence(
    dict1: Dict[str, torch.Tensor],
    dict2: Dict[str, torch.Tensor],
    atol: float = ATOL,
    rtol: float = RTOL,
) -> bool:
    if len(dict1) != len(dict2):
        return False

    for key in dict1:
        if not torch.allclose(dict1[key], dict2[key], atol=atol, rtol=rtol):
            return False
    return True


@torch.no_grad()
def reshape_parameter_gradient_to_module_matrix(
    module: nn.Module,
    module_name: str,
    gradient_dict: Dict[str, torch.Tensor],
    remove_gradient: bool = True,
) -> Optional[torch.Tensor]:
    if isinstance(module, nn.Linear):
        if module_name == "lm_head":
            # Edge case for small GPT model.
            return
        gradient_matrix = gradient_dict[module_name + ".weight"]
        if remove_gradient:
            del gradient_dict[module_name + ".weight"]
        if module_name + ".bias" in gradient_dict:
            gradient_matrix = torch.cat(
                (gradient_matrix, gradient_dict[module_name + ".bias"].unsqueeze(-1)),
                -1,
            )
            if remove_gradient:
                del gradient_dict[module_name + ".bias"]
    elif isinstance(module, nn.Conv2d):
        gradient_matrix = gradient_dict[module_name + ".weight"]
        if len(gradient_matrix.size()) == 5:
            gradient_matrix = gradient_matrix.view(gradient_matrix.size(0), gradient_matrix.size(1), -1)
        else:
            gradient_matrix = gradient_matrix.view(gradient_matrix.size(0), -1)
        if remove_gradient:
            del gradient_dict[module_name + ".weight"]
        if module_name + ".bias" in gradient_dict:
            gradient_matrix = torch.cat(
                [gradient_matrix, gradient_dict[module_name + ".bias"].unsqueeze(-1)],
                -1,
            )
            if remove_gradient:
                del gradient_dict[module_name + ".bias"]
    else:
        error_msg = f"Unsupported module type: {type(module)}. Only `nn.Linear` or `nn.Conv2d` are supported."
        raise UnsupportableModuleError(error_msg)
    return gradient_matrix
