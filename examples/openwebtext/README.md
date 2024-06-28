```bash
python analyze.py --factor_batch_size 32 \
    --train_batch_size 64 \
    --factor_strategy ekfac
```


```bash
torchrun --standalone --nnodes=1 --nproc-per-node=2 analyze.py --factor_batch_size 8 \
    --factor_strategy ekfac
```