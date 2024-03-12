<p align="center">
<a href="#"><img width="380" img src=".assets/kronfluence.svg" alt="Kronfluence Logo"/></a>
</p>

<p align="center">
    <a href="https://github.com/pomonam/kronfluence/LICENSE.md">
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

> **kronfluence** is a PyTorch-based library designed to compute [influence functions](https://arxiv.org/abs/1703.04730) using [Kronecker-factored Approximate Curvature (KFAC)](https://arxiv.org/abs/1503.05671) or [Eigenvalue-corrected KFAC (EKFAC)](https://arxiv.org/abs/1806.03884).
For a detailed description of the methodology, see the [**paper**](https://arxiv.org/abs/2308.03296) *Studying Large Language Model Generalization with Influence Functions*.

---

> [!WARNING]
> This library is under active development and has not reached its first stable release.

## Installation

> [!IMPORTANT]
> **Requirements:**
> - Python: Version 3.9 or later
> - PyTorch: Version 2.1 or later

To install the latest version of `kronfluence`, use the following `pip` command:

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


```python
import torch
import torch.nn.functional as F
```

(Placeholder for getting started content)

## Examples

(Placeholder for examples)

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
