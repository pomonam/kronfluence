# GPU Tests for Kronfluence

This folder contains various GPU tests for Kronfluence. Before running the tests, you need to prepare the 
baseline results by training an MNIST model and saving the results obtained with a single GPU:

```bash
python prepare_tests.py
```

### CPU Tests
To test if running on CPU yields the same result as the GPU, run:

```bash
python cpu_test.py
```

### DDP Tests
To test if running with Distributed Data Parallel (DDP) with 3 GPUs obtains the same result, run:

```bash
torchrun --nnodes=1 --nproc_per_node=3 ddp_test.py
```

### FSDP Tests
To test if running with Fully Sharded Data Parallel (FSDP) with 3 GPUs obtains the same result, run:

```bash
torchrun --nnodes=1 --nproc_per_node=3 fsdp_test.py
```

### torch.compile Tests
To test if running with `torch.compile` obtains the same result, run:

```bash
python compile_test.py
```

### AMP Tests
To test if running with automatic mixed precision (AMP) obtains the similar result, run:

```bash
python amp_test.py
```

### CPU Offload Test
To test if `cached_activation_cpu_offload` option is properly implemented, run:

```bash
pytest test_offload_cpu.py
```