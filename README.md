# Kronfluence

[![CI](https://github.com/mlcommons/algorithmic-efficiency/actions/workflows/CI.yml/badge.svg)](https://github.com/mlcommons/algorithmic-efficiency/actions/workflows/CI.yml)
[![Lint](https://github.com/mlcommons/algorithmic-efficiency/actions/workflows/linting.yml/badge.svg)](https://github.com/mlcommons/algorithmic-efficiency/actions/workflows/linting.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/mlcommons/algorithmic-efficiency/blob/main/LICENSE.md)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**kronfluence** is a PyTorch-based library designed to compute [influence functions](https://arxiv.org/abs/1703.04730) using [Kronecker-factored Approximate Curvature (KFAC)](https://arxiv.org/abs/1503.05671) or [Eigenvalue-corrected KFAC (EKFAC)](https://arxiv.org/abs/1806.03884).
For a detailed description of the methodology, see the [**paper**](https://arxiv.org/abs/2308.03296) *Studying Large Language Model Generalization with Influence Functions*.

> [!NOTE]  
> This library is an unofficial community implementation.

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

(Placeholder for getting started content)

## Examples

(Placeholder for examples)

## Contributing

Your contributions are welcome! See the [CONTRIBUTING](https://github.com/pomonam/kronfluence/blob/main/CONTRIBUTING.md) file to get started. For bug fixes, please submit a pull request without prior discussion. For proposing new features, examples, or extensions, kindly initiate a discussion through an issue before proceeding.

## License

This software is released under the Apache 2.0 License, as detailed in the [LICENSE](https://github.com/pomonam/kronfluence/blob/main/LICENSE) file.
