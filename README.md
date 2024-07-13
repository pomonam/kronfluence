<p align="center">
<a href="#"><img width="380" img src="https://raw.githubusercontent.com/pomonam/kronfluence/main/.assets/kronfluence.svg" alt="Kronfluence"/></a>
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
    <a href="https://github.com/pomonam/kronfluence/actions/workflows/linting.yml">
        <img alt="Linting" src="https://github.com/pomonam/kronfluence/actions/workflows/linting.yml/badge.svg">
    </a>
    <a href="https://github.com/astral-sh/ruff">
        <img alt="Ruff" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json">
    </a>
</p>

---

> **Kronfluence** is a PyTorch package designed to compute [influence functions](https://arxiv.org/abs/1703.04730) using [Kronecker-factored Approximate Curvature (KFAC)](https://arxiv.org/abs/1503.05671) or [Eigenvalue-corrected KFAC (EKFAC)](https://arxiv.org/abs/1806.03884).
For detailed description of the methodology, see the [**paper**](https://arxiv.org/abs/2308.03296), *Studying Large Language Model Generalization with Influence Functions*.

---

## Installation

> [!IMPORTANT]
> **Requirements:**
> - Python: Version 3.9 or later
> - PyTorch: Version 2.1 or later

To install the latest stable version, use the following `pip` command:

```bash
pip install kronfluence
```

Alternatively, you can install directly from source:

```bash
git clone https://github.com/pomonam/kronfluence.git
cd kronfluence
pip install -e .
```

## Getting Started

Kronfluence supports influence computations on [`nn.Linear`](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) and [`nn.Conv2d`](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) modules. 
See the [**Technical Documentation**](https://github.com/pomonam/kronfluence/blob/main/DOCUMENTATION.md) page for a comprehensive guide.

**TL;DR** You need to prepare a trained model and datasets, and pass them into the `Analyzer` class.

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

# Define the task. See the Technical Documentation page for details.
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

Kronfluence supports various PyTorch features; the following table summarizes the supported features:

<div align="center">

| Feature                                                                                                                     | Supported |
|-----------------------------------------------------------------------------------------------------------------------------|:---------:|
| [Distributed Data Parallel (DDP)](https://pytorch.org/docs/master/generated/torch.nn.parallel.DistributedDataParallel.html) |     ✅    |
| [Automatic Mixed Precision (AMP)](https://pytorch.org/docs/stable/amp.html)                                                 |     ✅    |
| [Torch Compile](https://pytorch.org/docs/stable/generated/torch.compile.html)                                               |     ✅    |
| [Gradient Checkpointing](https://pytorch.org/docs/stable/checkpoint.html)                                                   |     ✅    |
| [Fully Sharded Data Parallel (FSDP)](https://pytorch.org/docs/stable/fsdp.html)                                                                                      |     ✅    |

</div>

The [examples](https://github.com/pomonam/kronfluence/tree/main/examples) folder contains several examples demonstrating how to use Kronfluence. 

## LogIX

While Kronfluence supports influence function computations on large-scale models like `Meta-Llama-3-8B-Instruct`, for those 
interested in running influence analysis on even larger models or with a large number of query data points, our
project [LogIX](https://github.com/logix-project/logix) may be worth exploring. It integrates with frameworks like 
[HuggingFace Trainer](https://huggingface.co/docs/transformers/en/main_classes/trainer) and [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) 
and is also compatible with many PyTorch features (DDP & FSDP & [DeepSpeed](https://github.com/microsoft/DeepSpeed)). 

## Contributing

Contributions are welcome! To get started, please review our [Code of Conduct](https://github.com/pomonam/kronfluence/blob/main/CODE_OF_CONDUCT.md). For bug fixes, please submit a pull request. 
If you would like to propose new features or extensions, we kindly request that you open an issue first to discuss your ideas.

### Setting Up Development Environment

To contribute to Kronfluence, you will need to set up a development environment on your machine. 
This setup includes installing all the dependencies required for linting and testing.

```bash
git clone https://github.com/pomonam/kronfluence.git
cd kronfluence
pip install -e ."[dev]"
```

### Style Testing

To maintain code quality and consistency, we run ruff and linting tests on pull requests. Before submitting a 
pull request, please ensure that your code adheres to our formatting and linting guidelines. The following commands will 
modify your code. It is recommended to create a Git commit before running them to easily revert any unintended changes.

Sort import orderings using [isort](https://pycqa.github.io/isort/):

```bash
isort kronfluence
```

Format code using [ruff](https://docs.astral.sh/ruff/):

```bash
ruff format kronfluence
```

To view all [pylint](https://www.pylint.org/) complaints, run the following command:

```bash
pylint kronfluence
```

Please address any reported issues before submitting your PR.

## Acknowledgements

[Omkar Dige](https://github.com/xeon27) contributed to the profiling, DDP, and FSDP utilities, and [Adil Asif](https://github.com/adil-a/) provided valuable insights and suggestions on structuring the DDP and FSDP implementations.
I also thank Hwijeen Ahn, Sang Keun Choe, Youngseog Chung, Minsoo Kang, Sophie Liao, Lev McKinney, Laura Ruis, Andrew Wang, and Kewen Zhao for their feedback.

## License

This software is released under the Apache 2.0 License, as detailed in the [LICENSE](https://github.com/pomonam/kronfluence/blob/main/LICENSE) file.
