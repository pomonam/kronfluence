# Kronfluence: Technical Documentation & FAQs

For a detailed description of the methodology, please refer to the [**paper**](https://arxiv.org/abs/2308.03296), *Studying Large Language Model Generalization with Influence Functions*.

## Requirements

Kronfluence has been tested and is compatible with the following versions of [PyTorch](https://pytorch.org/):
- Python: Version 3.9 or later
- PyTorch: Version 2.1 or later

## Supported Modules & Strategies

Kronfluence offers support for:
- Computing influence functions on selected PyTorch modules. Currently, we support `nn.Linear` and `nn.Conv2d`.
- Computing influence functions with several Hessian approximation strategies, including `identity`, `diagonal`, `kfac`, and `ekfac`.
- Computing pairwise and self-influence (with and without measurement) scores.

> [!NOTE]
> If there are additional modules you would like to see supported, please submit an issue on our GitHub repository.

---

## Step-by-Step Guide

See [UCI Regression example](https://github.com/pomonam/kronfluence/blob/main/examples/uci/) for the complete workflow and an interactive tutorial.

**Prepare Your Model and Dataset.** 
Before computing influence scores, you need to prepare the trained model and dataset. You can use any frameworks to 
train the model (e.g., [Pytorch Lightning](https://lightning.ai/docs/pytorch/stable/) or [HuggingFace Trainer](https://huggingface.co/docs/transformers/main_classes/trainer)); you just need to prepare the final model parameters.

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
This class contains information about the trained model and how influence scores will be computed:
(1) how to compute the training loss; (2) how to compute the measurable quantity (f(θ) in the [paper](https://arxiv.org/abs/2308.03296); see **Equation 5**);
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
        # TODO: [Optional] Complete this method.
        return None  # Compute influence scores on all available modules.

    def get_attention_mask(self, batch: Any) -> Optional[Union[Dict[str, torch.Tensor], torch.Tensor]]:
        # TODO: [Optional] Complete this method.
        return None  # Attention mask not used.
```

**Prepare Your Model for Influence Computations.**
Kronfluence wraps all supported modules within the model with [`TrackedModule`](https://github.com/pomonam/kronfluence/blob/main/kronfluence/module/tracked_module.py).
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
After calling `prepare_model`, you can create [DistributedDataParallel (DDP)](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) or 
[FullyShardedDataParallel (FSDP)](https://pytorch.org/docs/stable/fsdp.html) module.

**Set up the Analyzer and Fit Factors.** 
Initialize the `Analyzer` and run `fit_all_factors` to compute all factors that aim to approximate the Hessian 
(Gauss-Newton Hessian). The computed factors will be stored on disk.

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
Once the factors have been computed, you can compute pairwise and self-influence scores. When computing the scores,
you can specify the factor name you would like to use. 

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

You can organize all factors and scores for the specific model with `factors_name` and `scores_name`.

### FAQs

**What should I do if my model does not have any nn.Linear or nn.Conv2d modules?**
Currently, the implementation does not support influence computations for modules other than `nn.Linear` or `nn.Conv2d`.
Try rewriting the model so that it uses supported modules (as done for the `conv1d` module in the [GPT-2 example](https://github.com/pomonam/kronfluence/tree/documentation/examples/wikitext)).
Alternatively, you can create a subclass of `TrackedModule` to compute influence scores for your custom module.
If there are specific modules you would like to see supported, please submit an issue.

**How should I write task.tracked_modules?**
We recommend using all supported modules for influence computations. However, if you would like to compute influence scores
on subset of the modules (e.g., influence computations only on MLP layers for transformer or influence computation only on the last layer),
inspect `model.named_modules()` to determine what modules to use. You can specify the list of module names you want to analyze.

> [!TIP]
> `Analyzer.get_module_summary(model)` can be helpful in figuring out what modules to include.

> [!NOTE]
> If the embedding layer for transformers are defined with `nn.Linear`, you must write your own
> `task.tracked_modules` to avoid influence computations embedding matrices.

**How should I implement Task.compute_train_loss?**
Implement the loss function used to train the model. Note that the function should return 
the summed loss (over batches and tokens). 

**How should I implement Task.compute_measurement?**
It depends on the analysis you would like to perform. Influence functions approximate the [effect of downweighting/upweighting
a training data point on the query's measurable quantity](https://arxiv.org/abs/2209.05364). You can use the loss, [margin](https://arxiv.org/abs/2303.14186) (for classification), 
or [conditional log-likelihood](https://arxiv.org/abs/2308.03296) (for language modeling). Note that many influence functions implementation, by default, uses the loss.

**I encounter TrackedModuleNotFoundError when using DDP or FSDP.**
Make sure to call `prepare_model` before wrapping your model with DDP or FSDP. Calling `prepare_model` on DDP modules can
cause `TrackedModuleNotFoundError`.

**My model uses supported modules, but influence scores are not computed.**
Kronfluence uses [module hooks](https://pytorch.org/docs/stable/generated/torch.Tensor.register_hook.html) to compute factors and influence scores. For these to be tracked and computed,
the model's forward pass should directly call the module.

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

> [!WARNING]
> The default arguments assume the module is used only once during the forward pass.
> IIf your model shares parameters (e.g., the module is used in multiple places during the forward pass), set
> `shared_parameters_exist=True` in both `FactorArguments` and `ScoreArguments`.

**Why are there so many arguments?**
Kronfluence was originally developed to compute influence scores on large-scale models, which is why `FactorArguments` and `ScoreArguments` 
have many parameters to support these use cases. However, for most standard applications, the default argument values 
should suffice. Feel free to use the default settings unless you have specific requirements that necessitate customization.

**I get X error when fitting factors/computing scores.**
Please feel free to contact me by [filing an issue](https://github.com/pomonam/kronfluence/issues) or [through email](mailto:jbae@cs.toronto.edu).

---

## Configuring Factors with FactorArguments

```python
import torch
from kronfluence.arguments import FactorArguments

factor_args = FactorArguments(
    strategy="ekfac",  # Choose from "identity", "diagonal", "kfac", or "ekfac".
    use_empirical_fisher=False,
    distributed_sync_steps=1000,
    amp_dtype=None,

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
    shared_parameters_exist=False,
    per_sample_gradient_dtype=torch.float32,
    lambda_dtype=torch.float32,
)

# You can pass in the arguments when fitting the factors.
analyzer.fit_all_factors(factors_name="initial_factor", dataset=train_dataset, factor_args=factor_args)
```

You can change:
- `strategy`: Selects the Hessian approximation strategy (`identity`, `diagonal`, `kfac`, or `ekfac`).
- `use_empirical_fisher`: Determines whether to use the [empirical Fisher](https://arxiv.org/abs/1905.12558) (using actual labels from batch) 
instead of the true Fisher (using sampled labels from model's predictions). It is recommended to be `False`.
- `amp_dtype`: Selects the dtype for [automatic mixed precision (AMP)](https://pytorch.org/docs/stable/amp.html). Disables AMP if set to `None`.

### Fitting Covariance Matrices

`kfac` and `ekfac` require computing the uncentered activation and pre-activation pseudo-gradient covariance matrices. 
To fit covariance matrices, you can use `analyzer.fit_covariance_matrices`.

```python
# Fitting covariance matrices.
analyzer.fit_covariance_matrices(factors_name="initial_factor", dataset=train_dataset, factor_args=factor_args)
# Loading covariance matrices.
covariance_matrices = analyzer.load_covariance_matrices(factors_name="initial_factor")
```

This step corresponds to **Equation 16** in the paper. You can tune:
- `covariance_max_examples`: Controls the maximum number of data points for fitting covariance matrices. Setting it to `None`,
Kronfluence computes covariance matrices for all data points.
- `covariance_data_partition_size`: Number of data partitions to use for computing covariance matrices. 
For example, when `covariance_data_partition_size = 2`, the dataset is split into 2 chunks and covariance matrices 
are separately computed for each chunk. These chunked covariance matrices are later aggregated. This is useful with GPU preemption as intermediate 
covariance matrices will be saved in disk. It can be also helpful when launching multiple parallel jobs, where each GPU
can compute covariance matrices on some partitioned data (you can specify `target_data_partitions` in the parameter).
- `covariance_module_partition_size`: Number of module partitions to use for computing covariance matrices.
For example, when `covariance_module_partition_size = 2`, the module is split into 2 chunks and covariance matrices 
are separately computed for each chunk. This is useful when the available GPU memory is limited (e.g., the total 
covariance matrices cannot fit into GPU memory). However, this will require multiple iterations over the dataset and can be slow.
- `activation_covariance_dtype`: `dtype` for computing activation covariance matrices. You can also use `torch.bfloat16`
or `torch.float16`.
- `gradient_covariance_dtype`: `dtype` for computing pre-activation pseudo-gradient covariance matrices. You can also use `torch.bfloat16`
or `torch.float16`.

**Dealing with OOMs.** Here are some steps to fix Out of Memory (OOM) errors.
1. Try reducing the `per_device_batch_size` when fitting covariance matrices.
2. Try using lower precision for `activation_covariance_dtype` and `gradient_covariance_dtype`.
3. Try setting `covariance_module_partition_size > 1`.

### Performing Eigendecomposition

After computing the covariance matrices, `kfac` and `ekfac` require performing eigendecomposition.

```python
# Performing eigendecomposition.
analyzer.perform_eigendecomposition(factors_name="initial_factor", factor_args=factor_args)
# Loading eigendecomposition results (e.g., eigenvectors and eigenvalues).
eigen_factors = analyzer.load_eigendecomposition(factors_name="initial_factor")
```

This corresponds to **Equation 18** in the paper. You can tune:
- `eigendecomposition_dtype`: `dtype` for performing eigendecomposition. You can also use `torch.float32`,
but `torch.float64` is strongly recommended.

### Fitting Lambda Matrices

`ekfac` and `diagonal` require computing the Lambda (eigenvalue) matrices for all modules.

```python
# Fitting Lambda matrices.
analyzer.fit_lambda_matrices(factors_name="initial_factor", dataset=train_dataset, factor_args=factor_args)
# Loading Lambda matrices.
lambda_matrices = analyzer.load_lambda_matrices(factors_name="initial_factor")
```

This corresponds to **Equation 20** in the paper. You can tune:
- `lambda_max_examples`: Controls the maximum number of data points for fitting Lambda matrices.
- `lambda_data_partition_size`: Number of data partitions to use for computing Lambda matrices. 
- `lambda_module_partition_size`: Number of module partitions to use for computing Lambda matrices. 
- `cached_activation_cpu_offload`: Computing the per-sample-gradient requires saving the intermediate activation in memory.
You can set `cached_activation_cpu_offload=True` to cache these activations in CPU. This is helpful for dealing with OOMs, but will make the overall computation slower.
- `lambda_iterative_aggregate`: Whether to compute the Lambda matrices with for-loops instead of batched matrix multiplications.
This is helpful for reducing peak memory, as it avoids holding multiple copies of tensors with the same shape as the per-sample-gradient.
- `shared_parameters_exist`: Specifies whether the shared parameters exist in the forward pass.
- `per_sample_gradient_dtype`: `dtype` for computing per-sample-gradient. You can also use `torch.bfloat16`
or `torch.float16`.
- `lambda_dtype`: `dtype` for computing Lambda matrices. You can also use `torch.bfloat16`
or `torch.float16`.

**Dealing with OOMs.** Here are some steps to fix Out of Memory (OOM) errors.
1. Try reducing the `per_device_batch_size` when fitting Lambda matrices.
2. Try setting `lambda_iterative_aggregate=True` or `cached_activation_cpu_offload=True`. (Try out `lambda_iterative_aggregate=True` first.)
3. Try using lower precision for `lambda_dtype`. 
4. Try using `lambda_module_partition_size > 1`. 

### FAQs

**I get different factors each time I run the code.**
This is expected as we sample labels from the model's prediction when computing covariance and Lambda matrices.
Using `use_empirical_fisher=True` could make the process more deterministic. Moreover, different hardware might compute
different eigenvectors when performing eigendecomposition.

**How should I select the batch size?**
You can use the largest possible batch size that does not result in OOM. Typically, the batch size for fitting Lambda
matrices should be smaller than that used for fitting covariance matrices. Furthermore, note that you should be getting similar results, regardless
of what batch size you use (different from training neural networks).

**Why are there so many arguments?**
Kronfluence was originally developed to compute influence scores on large-scale models, which is why `FactorArguments` 
have many parameters to support these use cases. There are some common `FactorArguments` in `common_factor_arguments.py`.

---

## Configuring Scores with ScoreArguments

```python
import torch
from kronfluence.arguments import ScoreArguments

score_args = ScoreArguments(
    damping=1e-08,
    cached_activation_cpu_offload=False,
    distributed_sync_steps=1000,
    amp_dtype=None,

    # More functionalities to compute influence scores.
    data_partition_size=1,
    module_partition_size=1,
    per_module_score=False,
    use_measurement_for_self_influence=False,

    # Configuration for query batching.
    query_gradient_rank=None,
    query_gradient_svd_dtype=torch.float32,
    num_query_gradient_aggregations=1,
    
    # Configuration for dtype.
    score_dtype=torch.float32,
    per_sample_gradient_dtype=torch.float32,
    precondition_dtype=torch.float32,
)
```

- `damping`: A damping factor for the damped inverse Hessian-vector product (iHVP). Uses a heuristic based on mean eigenvalues 
`(0.1 x mean eigenvalues)` if `None`, as done in [this paper](https://arxiv.org/abs/2308.03296).
- `cached_activation_cpu_offload`: Whether to offload cached activations to CPU.
- `amp_dtype`: Selects the dtype for [automatic mixed precision (AMP)](https://pytorch.org/docs/stable/amp.html). Disables AMP if set to `None`.
- `data_partition_size`: Number of data partitions for computing influence scores.
- `module_partition_size`: Number of module partitions for computing influence scores.
- `per_module_score`: Whether to return a per-module influence scores. Instead of summing over influences across
all modules, this will keep track of intermediate module-wise scores. 
- `per_token_score`: Whether to return a per-token influence scores. Instead of summing over influence scores across
all tokens, this will keep track of influence scores for each token. Note that this is only supported for Transformer-based models (language modeling).
- `query_gradient_rank`: The rank for the query batching (low-rank approximation to the query gradient; see **Section 3.2.2**). If `None`, no query batching will be used.
- `query_gradient_svd_dtype`: `dtype` for performing singular value decomposition (SVD) for query batch. You can also use `torch.float64`.
- `num_query_gradient_aggregations`: Number of query gradients to aggregate over. For example, when `num_query_gradient_aggregations = 2` with 
`query_batch_size = 16`, a total of 32 query gradients will be stored in memory when computing dot products with training gradients.
- `use_measurement_for_self_influence`: Whether to use the measurement (instead of the loss) when computing self-influence scores.
- `score_dtype`: `dtype` for computing influence scores. You can use `torch.bfloat16` or `torch.float16`.
- `per_sample_gradient_dtype`: `dtype` for computing per-sample-gradient. You can use `torch.bfloat16` or `torch.float16`.
- `precondition_dtype`: `dtype` for performing preconditioning. You can use `torch.bfloat16` or `torch.float16`,
but `torch.float32` is recommended.

### Computing Influence Scores

To compute pairwise influence scores (**Equation 5** in the paper), you can run:

```python
# Computing pairwise influence scores.
analyzer.compute_pairwise_scores(scores_name="pairwise", factors_name="ekfac", score_args=score_args)
# Loading pairwise influence scores.
scores = analyzer.load_pairwise_scores(scores_name="pairwise")
```

To compute self-influence scores (see **Section 5.4** from [this paper](https://arxiv.org/pdf/1703.04730.pdf)), you can run:

```python
# Computing self-influence scores.
analyzer.compute_self_scores(scores_name="self", factors_name="ekfac", score_args=score_args)
# Loading self-influence scores.
scores = analyzer.load_self_scores(scores_name="self")
```

By default, self-influence score computations only use the loss function for gradient calculations. 
In this case, the method returns a vector of size `len(train_dataset)`, where each value corresponds 
to `g_l^T ⋅ H^{-1} ⋅ g_l`. Here, `g_l` denotes the gradient of the loss function with respect to the model parameters, 
and `H^{-1}` represents the inverse Hessian matrix. If you want to use the measurement function instead of the loss function 
for self-influence calculations, set `use_measurement_for_self_influence=True`. In this case, each value in the returned 
vector will correspond to `g_m^T ⋅ H^{-1} ⋅ g_l`, where `g_m` is the gradient of the measurement function with respect to the model parameters.

**Dealing with OOMs.** Here are some steps to fix Out of Memory (OOM) errors.
1. Try reducing the `per_device_query_batch_size` or `per_device_train_batch_size`.
2. Try setting `cached_activation_cpu_offload=True`.
3. Try using lower precision for `per_sample_gradient_dtype` and `score_dtype`. 
4. Try setting `query_gradient_rank > 1`. The recommended values are `16`, `32`, `64`, `128`, and `256`. Note that query
batching is only supported for computing pairwise influence scores, not self-influence scores.
5. Try setting `module_partition_size > 1`.

### FAQs

**How should I choose a damping term?**
When setting the damping term, both `1e-08` and `None` are reasonable choices. The optimal value may depend on your 
specific workload. Another heuristic, suggested in [this paper](https://arxiv.org/abs/2405.12186), is to use `10 * learning_rate * num_iterations` when the model 
was trained using SGD with a momentum of 0.9. In practice, I have observed that the damping term does not significantly 
affect the final results as long as it is not too large (e.g., `1e-01`). Feel free to experiment with different values within a 
reasonable range to find what works best for your use case.

**Influence scores are very large in magnitude.**
Ideally, influence scores need to be divided by the total number of training data points. However, the code does
not normalize the scores. If you would like, you can divide the scores with the total number of data points (or tokens) used to
train the model.

## References

1. [Studying Large Language Model Generalization with Influence Functions](https://arxiv.org/abs/2308.03296). Roger Grosse, Juhan Bae, Cem Anil, et al. Tech Report, 2023.
2. [If Influence Functions are the Answer, Then What is the Question?](https://arxiv.org/abs/2209.05364). Juhan Bae, Nathan Ng, Alston Lo, Marzyeh Ghassemi, Roger Grosse. NeurIPS, 2022.
3. [TRAK: Attributing Model Behavior at Scale](https://arxiv.org/abs/2303.14186). Sung Min Park, Kristian Georgiev, Andrew Ilyas, Guillaume Leclerc, Aleksander Madry. ICML, 2023.
4. [Understanding Black-box Predictions via Influence Functions](https://arxiv.org/abs/1703.04730). Pang Wei Koh, Percy Liang. ICML, 2017.
5. [Optimizing Neural Networks with Kronecker-factored Approximate Curvature](https://arxiv.org/abs/1503.05671). James Martens, Roger Grosse. Tech Report, 2015.
5. [Fast Approximate Natural Gradient Descent in a Kronecker-factored Eigenbasis](https://arxiv.org/abs/1806.03884). Thomas George, César Laurent, Xavier Bouthillier, Nicolas Ballas, Pascal Vincent. NeurIPS, 2018.
6. [Training Data Attribution via Approximate Unrolled Differentiation](https://arxiv.org/abs/2405.12186). Juhan Bae, Wu Lin, Jonathan Lorraine, Roger Grosse. Preprint, 2024.