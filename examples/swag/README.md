# SWAG & RoBERTa Example

This directory demonstrates fine-tuning RoBERTa on the SWAG dataset and computing influence scores. The implementation is inspired by [this HuggingFace example](https://github.com/huggingface/transformers/tree/main/examples/pytorch/multiple-choice) and showcases how to define `post_process_per_sample_gradient`.
Install the required packages:

```bash
pip install -r requirements.txt
```

## Training

To fine-tune RoBERTa on the SWAG dataset, run the following command:

```bash
python train.py --checkpoint_dir ./checkpoints \
    --train_batch_size 16 \
    --eval_batch_size 32 \
    --learning_rate 1e-05 \
    --weight_decay 0.001 \
    --num_train_epochs 3 \
    --seed 1004
```

The final checkpoint will be saved in the `./checkpoints` directory.

## Computing Pairwise Influence Scores

To calculate pairwise influence scores on 2000 query data points using `ekfac`, run:

```bash
python analyze.py --factor_batch_size 128 \
    --query_batch_size 100 \
    --train_batch_size 64 \
    --checkpoint_dir ./checkpoints \
    --factor_strategy ekfac
```

Alternative options for `factor_strategy` include `identity`, `diagonal`, and `kfac`. 
On an A100 (80GB), computing the pairwise scores (including EKFAC factors) takes approximately 10 hours:

```
----------------------------------------------------------------------------------------------------------------------------------
|  Action                       |  Mean duration (s)    |  Num calls            |  Total time (s)       |  Percentage %         |
----------------------------------------------------------------------------------------------------------------------------------
|  Total                        |  -                    |  11                   |  3.5124e+04           |  100 %                |
----------------------------------------------------------------------------------------------------------------------------------
|  Compute Pairwise Score       |  2.9384e+04           |  1                    |  2.9384e+04           |  83.656               |
|  Fit Lambda                   |  3578.7               |  1                    |  3578.7               |  10.189               |
|  Fit Covariance               |  2143.9               |  1                    |  2143.9               |  6.1036               |
|  Perform Eigendecomposition   |  10.213               |  1                    |  10.213               |  0.029078             |
|  Save Eigendecomposition      |  3.4398               |  1                    |  3.4398               |  0.0097933            |
|  Save Covariance              |  2.5179               |  1                    |  2.5179               |  0.0071684            |
|  Save Pairwise Score          |  1.2982               |  1                    |  1.2982               |  0.0036959            |
|  Save Lambda                  |  0.68226              |  1                    |  0.68226              |  0.0019424            |
|  Load All Factors             |  0.013627             |  1                    |  0.013627             |  3.8797e-05           |
|  Load Eigendecomposition      |  0.0088496            |  1                    |  0.0088496            |  2.5195e-05           |
|  Load Covariance              |  0.008222             |  1                    |  0.008222             |  2.3408e-05           |
----------------------------------------------------------------------------------------------------------------------------------
```

For more efficient computation, use half-precision:

```bash
python analyze.py --factor_batch_size 128 \
    --query_batch_size 100 \
    --train_batch_size 128 \
    --use_half_precision \
    --checkpoint_dir ./checkpoints \
    --factor_strategy ekfac
```

This reduces computation time to about 3 hours on an A100 (80GB) GPU.

```
----------------------------------------------------------------------------------------------------------------------------------
|  Action                       |  Mean duration (s)    |  Num calls            |  Total time (s)       |  Percentage %         |
----------------------------------------------------------------------------------------------------------------------------------
|  Total                        |  -                    |  11                   |  1.0935e+04           |  100 %                |
----------------------------------------------------------------------------------------------------------------------------------
|  Compute Pairwise Score       |  9576.4               |  1                    |  9576.4               |  87.578               |
|  Fit Lambda                   |  932.07               |  1                    |  932.07               |  8.524                |
|  Fit Covariance               |  411.81               |  1                    |  411.81               |  3.7661               |
|  Perform Eigendecomposition   |  10.623               |  1                    |  10.623               |  0.097145             |
|  Save Eigendecomposition      |  1.4735               |  1                    |  1.4735               |  0.013475             |
|  Save Covariance              |  1.2953               |  1                    |  1.2953               |  0.011846             |
|  Save Pairwise Score          |  0.66271              |  1                    |  0.66271              |  0.0060606            |
|  Save Lambda                  |  0.34022              |  1                    |  0.34022              |  0.0031114            |
|  Load All Factors             |  0.012041             |  1                    |  0.012041             |  0.00011012           |
|  Load Covariance              |  0.0079526            |  1                    |  0.0079526            |  7.2728e-05           |
|  Load Eigendecomposition      |  0.0076841            |  1                    |  0.0076841            |  7.0273e-05           |
----------------------------------------------------------------------------------------------------------------------------------
```

Query batching (low-rank approximation to the query gradient; see **Section 3.2.2** from the paper) can be used to compute influence scores with a larger query batch size:

```bash
python analyze.py --factor_batch_size 128 \
    --query_batch_size 100 \
    --train_batch_size 128 \
    --query_gradient_rank 32 \
    --use_half_precision \
    --checkpoint_dir ./checkpoints \
    --factor_strategy ekfac
```

On an A100 (80GB) GPU, it takes roughly 1 hour to compute the pairwise scores with query batching:

```
----------------------------------------------------------------------------------------------------------------------------------
|  Action                       |  Mean duration (s)    |  Num calls            |  Total time (s)       |  Percentage %         |
----------------------------------------------------------------------------------------------------------------------------------
|  Total                        |  -                    |  3                    |  2007.9               |  100 %                |
----------------------------------------------------------------------------------------------------------------------------------
|  Compute Pairwise Score       |  2007.2               |  1                    |  2007.2               |  99.966               |
|  Save Pairwise Score          |  0.66464              |  1                    |  0.66464              |  0.033102             |
|  Load All Factors             |  0.012345             |  1                    |  0.012345             |  0.00061484           |
----------------------------------------------------------------------------------------------------------------------------------
```

## Evaluating Linear Datamodeling Score

The `evaluate_lds.py` script computes the [linear datamodeling score (LDS)](https://arxiv.org/abs/2303.14186). It measures the LDS obtained by 
retraining the network 500 times with different subsets of the dataset (5 repeats and 100 masks). 
We obtain `0.33` LDS (`0.30` LDS with half precision and half precision + query batching).

```
Query Data Example:
 Option 0: <s>He looks disgusted and spits it out onto the plate.</s></s>He slides both hands around the crack.</s><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>
 Option 1: <s>He looks disgusted and spits it out onto the plate.</s></s>He passes someone to the bald guy.</s><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>
 Option 2: <s>He looks disgusted and spits it out onto the plate.</s></s>He picks up a piece of bread.</s><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>
 Option 3: <s>He looks disgusted and spits it out onto the plate.</s></s>He walks into the kitchen.</s><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>
 Label: 3

Top Influential Example:
 Option 0: <s>He lowers her hair back over the cut.</s></s>He lies fully clothed, still gazing at her scooter.</s><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>
 Option 1: <s>He lowers her hair back over the cut.</s></s>He bangs her head against her headrest.</s><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>
 Option 2: <s>He lowers her hair back over the cut.</s></s>He goes to the kitchen.</s><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>
 Option 3: <s>He lowers her hair back over the cut.</s></s>He gives him a sidelong look.</s><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>
 Label: 2
```