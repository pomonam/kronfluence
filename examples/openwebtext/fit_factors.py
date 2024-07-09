import argparse
import logging
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch import nn
from transformers import default_data_collator

from examples.openwebtext.pipeline import (
    construct_llama3,
    get_openwebtext_dataset,
)
from examples.openwebtext.task import LanguageModelingTask
from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.utils.common.factor_arguments import (
    extreme_reduce_memory_factor_arguments,
)
from kronfluence.utils.dataset import DataLoaderKwargs

BATCH_TYPE = Dict[str, torch.Tensor]

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True


def parse_args():
    parser = argparse.ArgumentParser(description="Influence factor computation on Openwebtext dataset.")

    parser.add_argument(
        "--factor_strategy",
        type=str,
        default="ekfac",
        help="Strategy to compute influence factors.",
    )
    parser.add_argument(
        "--factor_batch_size",
        type=int,
        default=8,
        help="Batch size for computing influence factors.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        default=False,
        help="Boolean flag to profile computations.",
    )
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    # Prepare the dataset.
    train_dataset = get_openwebtext_dataset()

    # Prepare the trained model.
    model = construct_llama3()

    # Define task and prepare model.
    task = LanguageModelingTask()
    model = prepare_model(model, task)

    accelerator = Accelerator()
    model = accelerator.prepare_model(model)

    analyzer = Analyzer(
        analysis_name="openwebtext",
        model=model,
        task=task,
        profile=args.profile,
    )
    # Configure parameters for DataLoader.
    dataloader_kwargs = DataLoaderKwargs(collate_fn=default_data_collator)
    analyzer.set_dataloader_kwargs(dataloader_kwargs)

    factors_name = args.factor_strategy
    factor_args = extreme_reduce_memory_factor_arguments(strategy=args.factor_strategy,
                                                         module_partitions=2,
                                                         dtype=torch.bfloat16)
    analyzer.fit_all_factors(
        factors_name=factors_name,
        dataset=train_dataset,
        per_device_batch_size=args.factor_batch_size,
        # per_device_batch_size=None,
        factor_args=factor_args,
        overwrite_output_dir=False,
    )


if __name__ == "__main__":
    main()
