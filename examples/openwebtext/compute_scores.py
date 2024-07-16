import argparse
import logging
from datetime import timedelta

import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from transformers import default_data_collator

from examples.openwebtext.pipeline import (
    construct_llama3,
    get_custom_dataset,
    get_openwebtext_dataset,
)
from examples.openwebtext.task import (
    LanguageModelingTask,
    LanguageModelingWithMarginMeasurementTask,
)
from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.utils.common.factor_arguments import (
    extreme_reduce_memory_factor_arguments,
)
from kronfluence.utils.common.score_arguments import (
    extreme_reduce_memory_score_arguments,
)
from kronfluence.utils.dataset import DataLoaderKwargs

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True


def parse_args():
    parser = argparse.ArgumentParser(description="Influence score computation on Openwebtext dataset.")

    parser.add_argument(
        "--factors_name",
        type=str,
        required=True,
        help="Name of the factor.",
    )
    parser.add_argument(
        "--scores_name",
        type=str,
        required=True,
        help="Name of the score.",
    )
    parser.add_argument(
        "--use_margin_for_measurement",
        action="store_true",
        default=False,
        help="Boolean flag whether to use margin for measurement.",
    )
    parser.add_argument(
        "--query_gradient_rank",
        type=int,
        default=-1,
        help="Rank for the low-rank query gradient approximation.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=8,
        help="Batch size for computing query gradients.",
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
    eval_dataset = get_custom_dataset()

    # Prepare the trained model.
    model = construct_llama3()

    # Define task and prepare model.
    task = LanguageModelingTask()
    if args.use_margin_for_measurement:
        task = LanguageModelingWithMarginMeasurementTask()
    model = prepare_model(model, task)

    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))  # 1.5 hours.
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    model = accelerator.prepare_model(model)

    analyzer = Analyzer(
        analysis_name="openwebtext",
        model=model,
        task=task,
        profile=args.profile,
    )
    # Configure parameters for DataLoader.
    dataloader_kwargs = DataLoaderKwargs(num_workers=4, collate_fn=default_data_collator, pin_memory=True)
    analyzer.set_dataloader_kwargs(dataloader_kwargs)

    rank = args.query_gradient_rank if args.query_gradient_rank != -1 else None
    # We set the damping term used for LLMs.
    score_args = extreme_reduce_memory_score_arguments(
        damping_factor=None, module_partitions=1, query_gradient_low_rank=rank, dtype=torch.bfloat16
    )
    score_args.query_gradient_accumulation_steps = 10
    # We can invest some time in getting more accurate SVD results.
    score_args.use_full_svd = True
    score_args.precondition_dtype = torch.float32
    score_args.per_sample_gradient_dtype = torch.float32
    analyzer.compute_pairwise_scores(
        scores_name=args.scores_name,
        score_args=score_args,
        factors_name=args.factors_name,
        query_dataset=eval_dataset,
        train_dataset=train_dataset,
        per_device_query_batch_size=1,
        per_device_train_batch_size=args.train_batch_size,
        overwrite_output_dir=True,
    )
    scores = analyzer.load_pairwise_scores(args.scores_name)["all_modules"]
    logging.info(f"Scores shape: {scores.shape}")


if __name__ == "__main__":
    main()
