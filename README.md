<p align="center">
<a href="#"><img width="380" img src=".assets/kronfluence.svg" alt="Kronfluence"/></a>
</p>

<p align="center">
    <a href="https://github.com/pomonam/kronfluence/blob/main/LICENSE">
        <img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg">
    </a>
    <a href="https://github.com/mlcommons/algorithmic-efficiency/actions/workflows/CI.yml">
        <img alt="CI" src="https://github.com/mlcommons/algorithmic-efficiency/actions/workflows/CI.yml/badge.svg">
    </a>
    <a href="https://github.com/mlcommons/algorithmic-efficiency/actions/workflows">
        <img alt="Linting" src="https://github.com/mlcommons/algorithmic-efficiency/actions/workflows/linting.yml/badge.svg">
    </a>
    <a href="https://github.com/astral-sh/ruff">
        <img alt="Ruff" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json">
    </a>
</p>

---

> **Kronfluence** is a PyTorch-based library designed to compute [influence functions](https://arxiv.org/abs/1703.04730) using [Kronecker-factored Approximate Curvature (KFAC)](https://arxiv.org/abs/1503.05671) or [Eigenvalue-corrected KFAC (EKFAC)](https://arxiv.org/abs/1806.03884).
For a detailed description of the methodology, see the [**paper**](https://arxiv.org/abs/2308.03296) *Studying Large Language Model Generalization with Influence Functions*.

---

> [!WARNING]
> This library is under active development and has not reached its first stable release.

## Installation

> [!IMPORTANT]
> **Requirements:**
> - Python: Version 3.9 or later
> - PyTorch: Version 2.1 or later

To install the latest version, use the following `pip` command:

```bash
pip install kronfluence
```

Alternatively, you can install the library directly from the source:

```bash
git clone https://github.com/pomonam/kronfluence.git
cd kronfluence
pip install -e .
```

## Getting Started

Kronfluence currently supports influence computations on `nn.Linear` and `nn.Conv2d` modules.
It also supports several other Hessian approximation techniques: `identity`, `diagonal`, `KFAC`, and `EKFAC`.
The implementation is compatible with [Distributed Data Parallel (DDP)](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html), 
[Fully Sharded Data Parallel (FSDP)](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html), and [`torch.compile`](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html).
See [DOCUMENTATION.md](https://github.com/pomonam/kronfluence/blob/main/DOCUMENTATION.md) for detailed description on how to configure the experiment.

### Examples

The [examples](https://github.com/pomonam/kronfluence/tree/main/examples) folder contains several examples on how to use Kronfluence.

**TL;DR:** You need to prepare the trained model and datasets, and pass them into the `Analyzer`.

```python
import torch
import torchvision
from torch import nn

from kronfluence.analyzer import Analyzer, prepare_model

# Define the model and load the trained model weights.
model = torch.nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 1024, bias=True),
    nn.ReLU(),
    nn.Linear(1024, 1024, bias=True),
    nn.ReLU(),
    nn.Linear(1024, 1024, bias=True),
    nn.ReLU(),
    nn.Linear(1024, 10, bias=True),
)
model.load_state_dict(torch.load("model_path.pth"))

# Load the dataset.
train_dataset = torchvision.datasets.MNIST(
    root="./data",
    download=True,
    train=True,
)
eval_dataset = torchvision.datasets.MNIST(
    root="./data",
    download=True,
    train=True,
)

# Initialize the task for MNIST with relevant loss and measurement function.
task = MnistTask()

# Prepare the model for influence computation with the specified task.
model = prepare_model(model=model, task=task)
analyzer = Analyzer(analysis_name="mnist", model=model, task=task)

# Fit all EKFAC factors for the given model on the training dataset.
analyzer.fit_all_factors(factors_name="ekfac", dataset=train_dataset)

# Compute all pairwise influence scores using the fitted factors.
analyzer.compute_pairwise_scores(
    scores_name="pairwise_scores",
    factors_name="ekfac",
    query_dataset=eval_dataset,
    train_dataset=train_dataset,
    per_device_query_batch_size=1024,
)

# Load the scores with dimension `len(eval_dataset) x len(train_dataset)`.
scores = analyzer.load_pairwise_scores(scores_name="pairwise_scoeres")
```

## Contributing

Your contributions are welcome! For bug fixes, please submit a pull request without prior discussion. For proposing 
new features, examples, or extensions, kindly start a discussion through an issue before proceeding.

### Setting Up Development Environment

To contribute, you will need to set up a development environment on your machine. 
This setup includes installing all the dependencies required for linting and testing.

```bash
git clone https://github.com/pomonam/kronfluence.git
cd kronfluence
pip install -e ."[dev]"
```

## License

This software is released under the Apache 2.0 License, as detailed in the [LICENSE](https://github.com/pomonam/kronfluence/blob/main/LICENSE) file.
