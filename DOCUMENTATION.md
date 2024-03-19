# Kronfluence: Technical Documentation & FAQs

## Requirements

Kronfluence has been tested on the following versions of [PyTorch](https://pytorch.org/):
- PyTorch >= 2.1;
- Python >= 3.9.

## Supported Modules & Strategies

Kronfluence supports:
- Computing influence functions on selected PyTorch modules. At the moment, we support `nn.Linear` and `nn.Conv2d`;
- Computing influence functions with several strategies: `identity`, `diagonal`, `KFAC`, and `EKFAC`;
- Computing pairwise and self-influence scores.

> [!NOTE]
> We are planning to support functionalities to ensemble influence scores in next release.

> [!NOTE]
> If there are specific modules you would like to see supported, please submit an issue.

---

## Step-by-Step Guide

See [UCI Regression example](https://github.com/pomonam/kronfluence/blob/main/examples/uci/) for the complete workflow and 
interactive tutorial.

**Prepare Your Model and Dataset.** 
Before computing influence scores, you need to prepare the trained model and dataset. Note that you can use any platforms to 
train the model (e.g., [Pytorch Lightning](https://lightning.ai/docs/pytorch/stable/) and [HuggingFace Trainer](https://huggingface.co/docs/transformers/main_classes/trainer)).
```python
...
# Get the model with the trained parameters.
model = construct_model()
# Get the training dataset.
train_dataset = prepare_train_dataset()
# Get the query dataset (e.g., validation/test dataset).
query_dataset = prepare_query_dataset()
...
```

**Define a Task.**
To compute influence scores, you need to define a [`Task`](https://github.com/pomonam/kronfluence/blob/main/kronfluence/task.py) class.
This class encapsulates information about the trained model and how influence scores will be computed:
(1) how to compute the training loss; (2) how to compute the measurement (f(θ) in the [paper](https://arxiv.org/abs/2308.03296));
(3) which modules to use for influence function computations; and (4) whether the model used [attention mask](https://huggingface.co/docs/transformers/en/glossary#attention-mask).

```python
from typing import Any, Dict, List, Optional, Union
import torch
from torch import nn
from kronfluence.task import Task

class YourTask(Task):
    def compute_train_loss(
        self,
        batch: Any,
        model: nn.Module,
        sample: bool = False,
    ) -> torch.Tensor:
        # TODO: Complete this method.

    def compute_measurement(
        self,
        batch: Any,
        model: nn.Module,
    ) -> torch.Tensor:
        # TODO: Complete this method.

    def tracked_modules(self) -> Optional[List[str]]:
        # TODO: Complete this method.
        return None  # Compute influence scores on all available modules.

    def get_attention_mask(self, batch: Any) -> Optional[Union[Dict[str, torch.Tensor], torch.Tensor]]:
        # TODO: Complete this method.
        return None  # No attention mask is used.
```

**Install TrackedModule into Your Model.**
Kronfluence wraps supported modules within the model with [`TrackedModule`](https://github.com/pomonam/kronfluence/blob/main/kronfluence/module/tracked_module.py).
This wrapper will be used for computing the factors and influence scores. Once your model is ready and the task is defined,
prepare your model with:

```python
from kronfluence.analyzer import prepare_model
...
task = YourTask()
model = prepare_model(model=model, task=task)
...
```

If you have specified specific module names in `Task.tracked_modules`, `TrackedModule` will only be installed for these modules.

**\[Optional\] Create a DDP and FSDP Module.** 
After calling `prepare_model`, you can create DDP or FSDP module or even use `torch.compile`.

**Set up the Analyzer and Fit Factors.** 
Initialize the `Analyzer` and execute `fit_all_factors` to compute all factors that aim to approximate the Hessian 
(or Gauss-Newton Hessian). The computed factors will be stored on disk.
```python
from kronfluence.analyzer import Analyzer
from kronfluence.utils.dataset import DataLoaderKwargs
...
analyzer = Analyzer(analysis_name="model_with_seed0", model=model, task=task)

# [Optional] Set up the parameters for the DataLoader.
dataloader_kwargs = DataLoaderKwargs(num_workers=4, pin_memory=True)
analyzer.set_dataloader_kwargs(dataloader_kwargs)

# Compute all factors.
analyzer.fit_all_factors(factors_name="initial_factor", dataset=train_dataset)
...
```

**Compute Influence Scores.** 
Once the factors have been computed, you can compute either pairwise or self-influence scores. When computing the scores,
you can specify the factor name you want to use. 
```python
...
scores = analyzer.compute_pairwise_scores(
    scores_name="initial_score",
    factors_name="initial_factor",
    query_dataset=query_dataset,
    train_dataset=train_dataset,
    per_device_query_batch_size=1024,
)
...
```

You can organize all factors and scores for the specific model with `factor_name` and `score_name`.

### FAQs

**What should I do if my model does not have any nn.Linear or nn.Conv2d modules?**
Currently, the implementation does not support influence computations for modules other than `nn.Linear` or `nn.Conv2d`.
Try rewriting the model so that they use supported modules (as done for the `conv1d` module in [GPT-2](https://github.com/pomonam/kronfluence/tree/documentation/examples/wikitext)).
Alternatively, you can create a subclass of `TrackedModule` to compute influence scores for your custom modules.
If there are specific modules you would like to see supported, please submit an issue.

**How should I write task.tracked_modules?**
We recommend using all supported modules for influence computations. However, if you would like to compute influence scores
on subset of the modules (e.g., influence computations only on MLP layers for transformer or influence computation only on the last layer),
inspect `model.named_modules()` to determine what modules to use. You can specify the list of module names you want to analyze.

> [!NOTE]
> If the embedding layer for transformers are defined with `nn.Linear`, you must write
> `task.tracked_modules` to avoid influence computations embedding matrices.

**How should I implement Task.compute_train_loss?**
Implement the loss function used to train the model. However, the function should return 
the summed loss (over batches and tokens) and should not include regularizations. 

**How should I implement Task.compute_measurement?**
It depends on the analysis you would like to perform. Influence functions approximate the [local effect of downweighting/upweighting
a data point on the query's measurable quantity](https://arxiv.org/abs/2209.05364). You can use the loss, [margin](https://arxiv.org/abs/2303.14186) (for classification), 
or [conditional log-likelihood](https://arxiv.org/abs/2308.03296) (for language modeling). 

**I encounter TrackedModuleNotFoundError while using DDP or FSDP.**
Ensure to call `prepare_model` before wrapping your model with DDP or FSDP. Calling `prepare_model` on DDP modules can
cause `TrackedModuleNotFoundError`.

**My model uses supported modules, but influence scores are not computed.**
Kronfluence uses module hooks to compute factors and influence scores. For these to be tracked and computed,
the model should directly call the module.
```python
import torch
from torch import nn
    ...
    self.linear = nn.Linear(8, 1, bias=True)
    ...
def forward(x: torch.Tensor) -> torch.Tensor:
    x = self.linear(x)  # This works 😊
    x = self.linear.weight @ x + self.linear.bias  # This does not work 😞
```

**I get X error when fitting factors/computing scores.**
Please feel free to contact us by [filing an issue](https://github.com/pomonam/kronfluence/issues) or [through email](mailto:jbae@cs.toronto.edu).

---

## Configuring Factors with FactorArguments

The `FactorArguments` allows configuring settings for computing influence factors.

```python
import torch
from kronfluence.arguments import FactorArguments

factor_args = FactorArguments(
    strategy="ekfac",  # Choose from "identity", "diagonal", "KFAC", or "EKFAC".
    use_empirical_fisher=False,
    immediate_gradient_removal=False,
    ignore_bias=False,

    # Settings for covariance matrix fitting.
    covariance_max_examples=100_000,
    covariance_data_partition_size=1,
    covariance_module_partition_size=1,
    activation_covariance_dtype=torch.float32,
    gradient_covariance_dtype=torch.float32,
    
    # Settings for Eigendecomposition.
    eigendecomposition_dtype=torch.float64,
    
    # Settings for Lambda matrix fitting.
    lambda_max_examples=100_000,
    lambda_data_partition_size=1,
    lambda_module_partition_size=1,
    lambda_iterative_aggregate=False,
    cached_activation_cpu_offload=False,
    lambda_dtype=torch.float32,
)

# You can pass in the arguments when fitting the factors.
analyzer.fit_all_factors(factors_name="initial_factor", dataset=train_dataset, factor_args=factor_args)
```

You can change:
- `strategy`:  Selects the preconditioning strategy (`identity`, `diagonal`, `KFAC`, or `EKFAC`).
- `use_empirical_fisher`: Determines whether to approximate the [empirical Fisher](https://arxiv.org/abs/1905.12558) (using actual labels from batch) 
instead of the true Fisher (using sampled labels from model's predictions). It is recommended to be `False`.
However, `use_empirical_fisher = True` does not require implementing the case `sample = True`
in `Task.compute_train_loss` and can be used as an approximation if you are unsure how to implement when `sample = True`.
- `immediate_gradient_removal`: Specifies whether to instantly set `param.grad = None` within module hooks. Generally,
recommended to be `False`, as it requires installing additional hooks. This should not affect the fitted factors, but
can potentially reduce peak memory.
- `ignore_bias`: Specifies whether to ignore factor computations on bias.

### Fitting Covariance Matrices

`KFAC` and `EKFAC` require computing the activation and pseudo-gradient covariance matrices. 
To fit covariance matrices, you can use `analyzer.fit_covariance_matrices`.
```python
# Fitting covariance matrices.
analyzer.fit_covariance_matrices(factors_name="initial_factor", dataset=train_dataset, factor_args=factor_args)
# Loading covariance matrices.
covariance_matrices = analyzer.load_covariance_matrices(factors_name="initial_factor")
```

You can tune:
- `covariance_max_examples`: Controls the maximum number of data points for fitting covariance matrices. Setting it to `None`,
Kronfluence computes covariance matrices for all data points.
- `covariance_data_partition_size`: Number of data partitions to use for computing covariance matrices. 
For example, when `covariance_data_partition_size = 2`, the dataset is split into 2 chunks and covariance matrices 
are separately computed for each chunk. These chunked covariance matrices are later aggregated. This is useful with GPU preemption as intermediate 
covariance matrices will be saved in disk. It can be also helpful when launching multiple parallel jobs, where each GPU
can compute covariance matrices on some partitioned data (You can specify `target_data_partitions` in the parameter).
This should not affect the quality of the fitted factors.
- `covariance_module_partition_size`: Number of module partitions to use for computing covariance matrices.
For example, when `covariance_module_partition_size = 2`, the module is split into 2 chunks and covariance matrices 
are separately computed for each chunk. This is useful when the available GPU memory is limited (e.g., the total 
covariance matrices cannot fit into memory). However, this will do multiple iterations over the dataset and can be slow.
This should not affect the quality of the fitted factors.
- `activation_covariance_dtype`: `dtype` for computing activation covariance matrices. You can also use `torch.bfloat16`
or `torch.float16`.
- `gradient_covariance_dtype`: `dtype` for computing activation covariance matrices. You can also use `torch.bfloat16`
or `torch.float16`.

**Dealing with OOMs.** Here are some steps to fix Out of memory (OOM) errors.
1. Try reducing the `per_device_batch_size` when fitting covariance matrices.
2. Try using lower precision for `activation_covariance_dtype` and `gradient_covariance_dtype`.
3. Try setting `immediate_gradient_removal=True`.
4. Try setting `covariance_module_partition_size > 1`.


### Performing Eigendecomposition

After computing the covariance matrices, `KFAC` and `EKFAC` require performing Eigendecomposition.

```python
# Performing Eigendecomposition.
analyzer.perform_eigendecomposition(factors_name="initial_factor", factor_args=factor_args)
# Loading Eigendecomposition results.
eigen_factors = analyzer.load_eigendecomposition(factors_name="initial_factor")
```

You can tune:
- `eigendecomposition_dtype`: `dtype` for performing Eigendecomposition. You can also use `torch.float32`,
but `torch.float64` is recommended.

### Fitting Lambda Matrices

`EKFAC` and `diagonal` require computing the Lambda matrices for all modules.

```python
# Fitting Lambda matrices.
analyzer.fit_lambda_matrices(factors_name="initial_factor", dataset=train_dataset, factor_args=factor_args)
# Loading Lambda matrices.
lambda_matrices = analyzer.load_lambda_matrices(factors_name="initial_factor")
```

You can tune:
- `lambda_max_examples`: Controls the maximum number of data points for fitting Lambda matrices.
- `lambda_data_partition_size`: Number of data partitions to use for computing Lambda matrices. 
- `lambda_module_partition_size`: Number of module partitions to use for computing Lambda matrices. 
- `cached_activation_cpu_offload`: Computing the per-sample-gradient requires saving the intermediate activation in memory.
You can set `cached_activation_cpu_offload=True` to cache these activations in CPU.
- `lambda_iterative_aggregate`: Whether to compute the Lambda matrices with for-loop instead of batched matrix multiplications.
This is helpful for reducing peak memory, as it avoids holding multiple copies of tensors with the same shape as the per-sample-gradient.
- `lambda_dtype`: `dtype` for computing Lambda matrices. You can also use `torch.bfloat16`
or `torch.float16`.


**Dealing with OOMs.** Here are some steps to fix Out of memory (OOM) errors.
1. Try reducing the `per_device_batch_size` when fitting Lambda matrices.
2. Try setting `lambda_iterative_aggregate=True` or `cached_activation_cpu_offload=True`.
3. Try using lower precision for `lambda_dtype`. 
4. Try setting `immediate_gradient_removal=True`.
5. Try using `lambda_module_partition_size > 1`. 

### FAQs

**I get different factors each time I run the code.**
This is expected as we sample labels from the model's prediction when computing covariance and Lambda matrices.
Using `use_empirical_fisher=True` could make the process more deterministic. Moreover, different hardware might compute
different eigenvectors when performing Eigendecomposition.

**How should I select the batch size?**
You can use the largest possible batch size that does not result in OOM. Typically, the batch size for fitting Lambda
matrices should be smaller than that used for fitting covariance matrices.

---

### Score Configuration

The `ScoreArguments` allows configuring settings for computing influence scores.

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


### FAQs

**How should I select the batch size?**
You can use the largest possible batch size that does not result in OOM.

**Influence scores are very large in magnitude**
Ideally, influence scores need to be divided by the total number of training data points. However, the code does
not normalize the scores.

## References

1. 