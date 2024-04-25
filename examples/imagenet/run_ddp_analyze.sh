#!/bin/bash
#SBATCH --partition=a40
#SBATCH --time=2:00:00
#SBATCH --mem=0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --output=./examples/imagenet/imagenet.%j.out
#SBATCH --error=./examples/imagenet/imagenet.%j.err

nvidia-smi

export PYTHONPATH=/fs01/home/odige/kronfluence
source /projects/aieng/public/kronfluence_env/bin/activate

# (while true; do nvidia-smi; top -b -n 1 | head -20; sleep 2; done) &

torchrun --standalone --nnodes=1 --nproc-per-node=4 \
    examples/imagenet/ddp_analyze.py \
        --dataset_dir /fs01/datasets/imagenet \
        --query_gradient_rank -1 \
        --factor_batch_size 2 \
        --query_batch_size 8 \
        --train_batch_size 8 \
        --factor_strategy ekfac \
        --profile