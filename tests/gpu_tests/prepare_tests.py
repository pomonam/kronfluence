# pylint: skip-file

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.arguments import FactorArguments, ScoreArguments
from tests.gpu_tests.pipeline import GpuTestTask, construct_test_mlp, get_mnist_dataset

# Pick difficult cases where the dataset is not perfectly divisible by batch size.
TRAIN_INDICES = 5_003
QUERY_INDICES = 51


def train() -> None:
    assert torch.cuda.is_available()
    device = torch.device("cuda")
    train_dataset = get_mnist_dataset(split="train", data_path="data")
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=128,
        shuffle=True,
        drop_last=True,
    )
    model = construct_test_mlp().to(device=device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03)

    model.train()
    for epoch in range(5):
        total_loss = 0
        with tqdm(train_dataloader, unit="batch") as tepoch:
            for batch in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                logits = model(inputs)
                loss = F.cross_entropy(logits, labels)
                total_loss += loss.detach().float()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                tepoch.set_postfix(loss=total_loss.item() / len(train_dataloader))

    model.eval()
    eval_dataset = get_mnist_dataset(split="valid", data_path="data")
    eval_dataloader = DataLoader(
        dataset=eval_dataset,
        batch_size=1024,
        shuffle=False,
        drop_last=False,
    )
    total_loss = 0
    correct = 0
    for batch in eval_dataloader:
        with torch.no_grad():
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            loss = F.cross_entropy(logits, labels)
            preds = logits.argmax(dim=1, keepdim=True)
            correct += preds.eq(labels.view_as(preds)).sum().item()
            total_loss += loss.detach().float()

    print(
        f"Train loss: {total_loss.item() / len(eval_dataloader.dataset)} | "
        f"Train Accuracy: {100 * correct / len(eval_dataloader.dataset)}"
    )

    torch.save(model.state_dict(), "model.pth")


def run_analysis() -> None:
    assert torch.cuda.is_available()
    device = torch.device("cuda")
    model = construct_test_mlp().to(device=device)
    model.load_state_dict(torch.load("model.pth"))

    train_dataset = get_mnist_dataset(split="train", data_path="data")
    eval_dataset = get_mnist_dataset(split="valid", data_path="data")
    train_dataset = Subset(train_dataset, indices=list(range(TRAIN_INDICES)))
    eval_dataset = Subset(eval_dataset, indices=list(range(QUERY_INDICES)))

    task = GpuTestTask()
    model = model.double()
    model = prepare_model(model, task)

    analyzer = Analyzer(
        analysis_name="gpu_test",
        model=model,
        task=task,
    )

    factor_args = FactorArguments(
        use_empirical_fisher=True,
        activation_covariance_dtype=torch.float64,
        gradient_covariance_dtype=torch.float64,
        lambda_dtype=torch.float64,
    )
    analyzer.fit_all_factors(
        factors_name="single_gpu",
        dataset=train_dataset,
        factor_args=factor_args,
        per_device_batch_size=512,
        overwrite_output_dir=True,
    )

    score_args = ScoreArguments(
        score_dtype=torch.float64,
        per_sample_gradient_dtype=torch.float64,
        precondition_dtype=torch.float64,
    )
    analyzer.compute_pairwise_scores(
        scores_name="single_gpu",
        factors_name="single_gpu",
        query_dataset=eval_dataset,
        train_dataset=train_dataset,
        per_device_query_batch_size=12,
        per_device_train_batch_size=512,
        score_args=score_args,
        overwrite_output_dir=True,
    )
    analyzer.compute_self_scores(
        scores_name="single_gpu",
        factors_name="single_gpu",
        train_dataset=train_dataset,
        per_device_train_batch_size=512,
        score_args=score_args,
        overwrite_output_dir=True,
    )

    score_args = ScoreArguments(
        query_gradient_rank=32,
        score_dtype=torch.float64,
        per_sample_gradient_dtype=torch.float64,
        precondition_dtype=torch.float64,
    )
    analyzer.compute_pairwise_scores(
        scores_name="single_gpu_qb",
        factors_name="single_gpu",
        query_dataset=eval_dataset,
        train_dataset=train_dataset,
        per_device_query_batch_size=12,
        per_device_train_batch_size=512,
        score_args=score_args,
        overwrite_output_dir=True,
    )


if __name__ == "__main__":
    train()
    run_analysis()
