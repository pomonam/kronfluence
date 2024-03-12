# Kronfluence: Technical Documentation & FAQs

---

## Requirements

Kronfluence has been tested on the following versions of PyTorch:
- PyTorch >= 2.1;
- Python >= 3.9.

## Supported Modules & Strategies

Kronfluence supports:
- Computing influence functions on supported modules. At the moment, we support `nn.Linear` and `nn.Conv2d` modules;
- Computing influence scores with several preconditioning strategies: `identity`, `diagonal`, `KFAC`, and `EKFAC`;
- Computing both pairwise and self-influence scores.

> [!NOTE]
> We are planning to support ensembling influence scores in our next release, which will support methods like [TracIn](https://arxiv.org/abs/2002.08484).

> [!NOTE]  
> If there are specific modules you would like to see supported, please submit an issue.

---
## Getting Started

**Prepare the trained model and dataset.** To compute influence scores, you need to first prepare the trained model.
You may use the model trained with any platforms. Also, you need to prepare the training and query dataset to compute
influence scores.

```python
model = construct_model()
train_dataset = prepare_train_dataset()
query_dataset = prepare_query_dataset()
```

**Define a Task.** You need to define a [`Task`](https://github.com/pomonam/kronfluence/blob/main/kronfluence/task.py) that have information about (1) how to compute the training loss,
(2) how to compute the measurement, (3) which modules to compute influence functions on, and (4) whether the model 
uses the attention mask.

```python
from typing import Any, Tuple, Optional, Union, List, Dict
from torch import nn
import torch

BATCH_DTYPE = Tuple[torch.Tensor, torch.Tensor]

class Task:
    def compute_train_loss(
        self,
        batch: Any,
        model: nn.Module,
        sample: bool = False,
    ) -> torch.Tensor:
        # TODO: Complete this method.
        pass

    def compute_measurement(
        self,
        batch: Any,
        model: nn.Module,
    ) -> torch.Tensor:
        # TODO: Complete this method.
        pass

    def influence_modules(self) -> Optional[List[str]]:
        # TODO: Complete this method.
        pass


    def get_attention_mask(self, batch: Any) -> Optional[Union[Dict[str, torch.Tensor], torch.Tensor]]:
        # TODO: Complete this method.
        pass
```

**Install `TrackedModule` to the trained model to compute various factors and influence scores.** 
Kronfluence installs a wrapper [`TrackedModule`](https://github.com/pomonam/kronfluence/blob/main/kronfluence/module/tracked_module.py) to 
the supported modules to compute influence scores. After preparing the model and defining the task, you can call the function:
```python
from kronfluence.analyzer import prepare_model

task = construct_task()

model = prepare_model(model, task)
```

**\[Optional\] Create a DDP and FSDP module or run `torch.compile`.** After installing the `TrackedModule`, you can 
create DDP or FSDP module or apply `torch.compile`.

**Initialize `Analyzer` and fit factors to approximate the Hessian.** Initialize the `Analyzer` and run `fit_all_factors` to fit all
factors. This should save the factors on disk.
```python
from kronfluence.analyzer import Analyzer

...
analyzer = Analyzer(analysis_name="mnist", model=model, task=task)
analyzer.fit_all_factors(factors_name="ekfac", dataset=train_dataset)
```

**Compute pairwise or self influence scores.** After fitting the factors, you can compute influence scores:
```python
scores = analyzer.compute_pairwise_scores(
    scores_name="pairwise_scores",
    factors_name="ekfac",
    query_dataset=eval_dataset,
    train_dataset=train_dataset,
    per_device_query_batch_size=1024,
)
```

### FAQs

**What if my model does not contain any `nn.Linear` or `nn.Conv2d` modules?**
At the moment, influence computations on other modules are not supported. You can rewrite the model to explicitly use
these modules (as done in the GPT-2 example), or write a subclass of `TrackedModule` to compute influence scores on
custom modules.

**How do I select the module names?** One way is to inspect `model.named_modules()` to determine what modules to use
for influence computation. By default, it will be computing influences on all supported modules.

**I am using DDP or FSDP, but having TrackedModuleNotFoundError**
Make sure to call `prepare_model` before wrapping your model with `DDP` or `FSDP`.

**My model contains supported modules, but scores are not computed.**
Kronfluence uses module hooks to implement influence functions. Hence, for the factors and scores to be correctly keep
tracked, they should call `nn.Module` directly. Make sure the model properly calls the module.
```python
from torch import nn

...
linear = nn.Linear(8, 1, bias=True)

# This works.
x = linear(x)

# This does not work.
x = linear.weight @ x + linear.bias
```

---

## Factor Configuration

The `FactorArguments` can be passed to configure factors.

```python
import torch
from kronfluence.arguments import FactorArguments

factor_args = FactorArguments(
    strategy="ekfac",
    use_empirical_fisher=False,
    immediate_gradient_removal=False,

    # Configurations for fitting covariance matrices.
    covariance_max_examples=100_000,
    covariance_data_partition_size=1,
    covariance_module_partition_size=1,
    activation_covariance_dtype=torch.float32,
    gradient_covariance_dtype=torch.float32,
    
    # Configuration for performing eigendecomposition.
    eigendecomposition_dtype=torch.float64,
    
    # Configuration for fitting Lambda matrices.
    lambda_max_examples=100_000,
    lambda_data_partition_size=1,
    lambda_module_partition_size=1,
    lambda_iterative_aggregate=False,
    cached_activation_cpu_offload=False,
    lambda_dtype=torch.float32,
)

# You can pass in the arguments when fitting the factors.
analyzer.fit_all_factors(factors_name="ekfac", dataset=train_dataset, factor_args=factor_args)
```

- `strategy`: `identity`, `diagonal`, `KFAC`, or `EKFAC`.
- `use_empirical_fisher`: Whether to use [empirical fisher](https://arxiv.org/abs/1905.12558) (using labels from batch) instead of true Fisher (using 
sampled labels).
- `immediate_gradient_removal`: Whether to immediately remove `param.grad` within a hook. This should be set to 
`False` in most cases.

### Fitting Covariance Matrices

`KFAC` and `EKFAC` requires computing the activation and pseudo-gradient covariance matrices for all modules. If you
only want to compute the covariance matrices, you can use:
```python
# Fitting covariance matrices.
analyzer.fit_covariance_matrices(factors_name="ekfac", dataset=train_dataset, factor_args=factor_args)
# Loading covariance matrices.
covariance_matrices = analyzer.load_covariance_matrices(factors_name="ekfac")
```

- `covariance_max_examples`: The number of data points used to fit covariance matrices. You can set this to `None`
to compute covariance factors with all training data points.
- `covariance_data_partition_size`: Number of data partitions for computing covariance matrices. 
For example, when `covariance_data_partition_size = 2`, the dataset is split into 2 chunks and covariance matrices 
are separately computed for each chunk. These chunked covariance matrices are later aggregated. This is useful with GPU preemption as intermediate 
covariance matrices will be saved in disk.
- `covariance_module_partition_size`: Number of module partitions for computing covariance matrices.
For example, when `covariance_module_partition_size = 2`, the module is split into 2 chunks and covariance matrices 
are separately computed for each chunk. This is useful when the available GPU memory is limited; the total 
covariance matrices cannot fit into memory. However, this will do multiple iterations over the dataset and can be slower.
- `activation_covariance_dtype`: `dtype` for computing activation covariance matrices. You can use `torch.bfloat16`
or `torch.float16`.
- `gradient_covariance_dtype`: `dtype` for computing activation covariance matrices. You can use `torch.bfloat16`
or `torch.float16`.


**Dealing with OOMs** Here are some steps to fix Out of memory (OOM) errors.
1. Try reducing the `per_device_batch_size` when fitting covariance matrices. (This should give same results.)
2. Try using lower precision for `activation_covariance_dtype` and `gradient_covariance_dtype`. (This gives different results.)
3. Try setting `immediate_gradient_removal=True`. (This should give same results.)
4. Try setting `covariance_module_partition_size > 1`. (This should give same results.)

**Too slow** Here are some steps if fitting covariance matrices are too slow.
1. Try increasing the `per_device_batch_size` when fitting covariance matrices. (This gives different results.)
2. Try using lower precision for `activation_covariance_dtype` and `gradient_covariance_dtype`. 
3. Try reducing `covariance_max_examples`.


### Performing Eigendecomposition

After computing the covariance matrices, `KFAC` and `EKFAC` require performing Eigendecomposition. If you
 want to compute perform Eigendecomposition, you can use:
```python
# Performing Eigendecomposition.
analyzer.perform_eigendecomposition(factors_name="ekfac", factor_args=factor_args)
# Loading Eigendecomposition results.
eigen_factors = analyzer.load_eigendecomposition(factors_name="ekfac")
```
- `eigendecomposition_dtype`: `dtype` for computing activation covariance matrices. You can use `torch.float32`,
but `torch.float64` is recommended.

### Fitting Lambda Matrices

`EKFAC` and `diagonal` require computing the Lambda matrices for all modules. If you
only want to compute the covariance matrices, you can use:
```python
# Fitting Lambda matrices.
analyzer.fit_lambda_matrices(factors_name="ekfac", dataset=train_dataset, factor_args=factor_args)
# Loading Lambda matrices.
lambda_matrices = analyzer.load_lambda_matrices()(factors_name="ekfac")
```

- `lambda_max_examples`: The number of data points used to fit covariance matrices. You can set this to `None`
to compute covariance factors with all training data points.
- `lambda_data_partition_size`: Number of data partitions for computing Lambda matrices
- `lambda_module_partition_size`: Number of module partitions for computing Lambda matrices.
- `cached_activation_cpu_offload`: Computing the per-sample-gradient requires saving the intermediate activation in memory.
You can set `cached_activation_cpu_offload=True` to save these activations in CPU.
- `lambda_iterative_aggregate`: Whether to aggregate the Lambda matrices with for-loop instead of matrix multiplications.
This is helpful to reduce peak memory, as avoids holding multiple copies of tensors with the same shape as the per-sample-gradient.
- `lambda_dtype`: `dtype` for computing Lambda matrices. You can use `torch.bfloat16`
or `torch.float16`.


**Dealing with OOMs** Here are some steps to fix Out of memory (OOM) errors.
1. Try reducing the `per_device_batch_size` when fitting Lambda matrices.
2. Try setting `lambda_iterative_aggregate=True` or `cached_activation_cpu_offload=True`.
3. Try using lower precision for `lambda_dtype`. 
4. Try setting `immediate_gradient_removal=True`.
5. Try setting `lambda_module_partition_size > 1`. 

**Too slow** Here are some steps if fitting covariance matrices are too slow.
1. Try increasing the `per_device_batch_size` when fitting Lambda matrices.
2. Try using lower precision for `lambda_dtype`. 
3. Try reducing `lambda_max_examples`. This does not neccesarily have to be the same as `covariance_max_examples`.


---

### Score Configuration

The `ScoreArguments` can be passed to configure scores.

```python
import torch
from kronfluence.arguments import ScoreArguments

score_args = ScoreArguments(
    damping=None,
    immediate_gradient_removal=False,

    data_partition_size=1,
    module_partition_size=1,
    per_module_score=False,
    
    # Configuration for query batching.
    query_gradient_rank=None,
    query_gradient_svd_dtype=torch.float64,
    
    cached_activation_cpu_offload=False,
    score_dtype=torch.float32,
    per_sample_gradient_dtype=torch.float32,
    precondition_dtype=torch.float32,
)
```

- `damping`: A damping factor for the damped matrix-vector product. Uses a heuristic based on mean eigenvalues 
(0.1 x mean eigenvalues) if None.
- `immediate_gradient_removal`: Whether to immediately remove `param.grad` within a hook. This should be set to 
`False` in most cases.
- `data_partition_size`: Number of data partitions for computing influence scores.
- `module_partition_size`: Number of module partitions for computing influence scores.
- `per_module_score`: Whether to return a per-module influence scores. Instead of summing over influences across
all modules, this will keep track of intermediate module-wise scores. 

- `query_gradient_rank`: The rank for the query batching. If `None`, no query batching will be used. The query batching
is helpful in cases where the number of training examples are large to reduce multiple recomputations.
- `query_gradient_svd_dtype`: `dtype` for performing singular value decomposition (SVD) for query batch. You can use `torch.float32`,
but `torch.float64` is recommended.

- `cached_activation_cpu_offload`: Computing the per-sample-gradient requires saving the intermediate activation in memory.
- `score_dtype`: `dtype` for computing influence scores. You can use `torch.bfloat16` or `torch.float16`.
- `per_sample_gradient_dtype`: `dtype` for computing per-sample-gradient. You can use `torch.bfloat16` or `torch.float16`.
- `precondition_dtype`: `dtype` for performing preconditioning. You can use `torch.bfloat16` or `torch.float16`,
but `torch.float32` is recommended.

### Computing Influence Scores

To compute pairwise influence scores, you can run:
```python
# Computing pairwise influence scores.
analyzer.compute_pairwise_scores(scores_name="pairwise", factors_name="ekfac", score_args=score_args)
# Loading pairwise influence scores.
scores = analyzer.load_pairwise_scores(scores_name="pairwise")
```

To compute self-influence scores, you can run:
```python
# Computing pairwise influence scores.
analyzer.compute_self_scores(scores_name="self", factors_name="ekfac", score_args=score_args)
# Loading pairwise influence scores.
scores = analyzer.load_pairwise_scores(scores_name="self")
```

**Dealing with OOMs** Here are some steps to fix Out of memory (OOM) errors.
1. Try reducing the `per_device_query_batch_size` or `per_device_train_batch_size`.
2. Try setting `cached_activation_cpu_offload=True`.
3. Try using lower precision for `per_sample_gradient_dtype` and `score_dtype`. 
4. Try setting `immediate_gradient_removal=True`.
6. Try setting `query_gradient_rank > 1`. The recommended values are `32`, `64`, `128`, and `256`.
5. Try setting `module_partition_size > 1`. 

**Too slow** Here are some steps if fitting covariance matrices are too slow.
1. Try increasing the `per_device_query_batch_size` or `per_device_train_batch_size`.
2. Try using lower precision for `per_sample_gradient_dtype` and `score_dtype`.

### FAQs

**What if my model does not contain any `nn.Linear` or `nn.Conv2d` modules?**


## Multi-GPU Support

