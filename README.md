<p align="center">
<a href="#"><img width="380" img src=".assets/kronfluence.svg" alt="Kronfluence"/></a>
</p>


<p align="center">
    <a href="https://pypi.org/project/kronfluence">
        <img alt="License" src="https://img.shields.io/pypi/v/kronfluence.svg?style=flat-square">
    </a>
    <a href="https://github.com/pomonam/kronfluence/blob/main/LICENSE">
        <img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg">
    </a>
    <a href="https://github.com/pomonam/kronfluence/actions/workflows/python-test.yml">
        <img alt="CI" src="https://github.com/pomonam/kronfluence/actions/workflows/python-test.yml/badge.svg">
    </a>
    <a href="https://github.com/mlcommons/algorithmic-efficiency/actions/workflows">
        <img alt="Linting" src="https://github.com/pomonam/kronfluence/actions/workflows/linting.yml/badge.svg">
    </a>
    <a href="https://github.com/astral-sh/ruff">
        <img alt="Ruff" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json">
    </a>
</p>

---

> **Kronfluence** is a research repository designed to compute [influence functions](https://arxiv.org/abs/1703.04730) using [Kronecker-factored Approximate Curvature (KFAC)](https://arxiv.org/abs/1503.05671) or [Eigenvalue-corrected KFAC (EKFAC)](https://arxiv.org/abs/1806.03884).
For a detailed description of the methodology, see the [**paper**](https://arxiv.org/abs/2308.03296) *Studying Large Language Model Generalization with Influence Functions*.

---

> [!WARNING]
> This repository is under active development and has not reached its first stable release.

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

Kronfluence supports influence computations on `nn.Linear` and `nn.Conv2d` modules. See the [**Technical Documentation**](https://github.com/pomonam/kronfluence/blob/main/DOCUMENTATION.md) 
page for a comprehensive guide.

### Learn More

The [examples](https://github.com/pomonam/kronfluence/tree/main/examples) folder contains several examples.
We plan to add more examples in the future. **TL;DR** You need to prepare the trained model and datasets, and pass them into `Analyzer`.

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

# Define the task.
task = MnistTask()

# Prepare the model for influence computation.
model = prepare_model(model=model, task=task)
analyzer = Analyzer(analysis_name="mnist", model=model, task=task)

# Fit all EKFAC factors for the given model.
analyzer.fit_all_factors(factors_name="my_factors", dataset=train_dataset)

# Compute all pairwise influence scores with the computed factors.
analyzer.compute_pairwise_scores(
    scores_name="my_scores",
    factors_name="my_factors",
    query_dataset=eval_dataset,
    train_dataset=train_dataset,
    per_device_query_batch_size=1024,
)

# Load the scores with dimension `len(eval_dataset) x len(train_dataset)`.
scores = analyzer.load_pairwise_scores(scores_name="my_scores")
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
