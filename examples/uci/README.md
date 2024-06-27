# UCI Regression Example

This directory contains scripts for training a regression model and conducting influence analysis using datasets from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/datasets). 
To get started, please install the necessary packages by running the following command:

```bash
pip install -r requirements.txt
```

## Training

To train a regression model on the `Concrete` dataset, run the following command:

```bash
python train.py --dataset_name concrete \
    --dataset_dir ./data \
    --output_dir ./checkpoints \
    --train_batch_size 32 \
    --eval_batch_size 1024 \
    --learning_rate 0.03 \
    --weight_decay 1e-5 \
    --num_train_epochs 20 \
    --seed 1004
```

This will train the model using the specified hyperparameters and save the trained checkpoint in the `./checkpoints` directory.

## Computing Pairwise Influence Scores

To compute pairwise influence scores using the `ekfac` factorization strategy, run the following command:

```bash
python analyze.py --dataset_name concrete \
    --dataset_dir ./data \
    --checkpoint_dir ./checkpoints \
    --factor_strategy ekfac
```

You can also use `identity`, `diagonal`, and `kfac` for `factor_strategy`. 
To measure the wall-clock time of computing influence scores, you can enable the `profile` flag:

```
----------------------------------------------------------------------------------------------------------------------------------
|  Action                    	|  Mean duration (s)	|  Num calls      	|  Total time (s) 	|  Percentage %   	|
----------------------------------------------------------------------------------------------------------------------------------
|  Total                     	|  -              	|  11             	|  0.35452        	|  100 %          	|
----------------------------------------------------------------------------------------------------------------------------------
|  Compute Pairwise Score    	|  0.13146        	|  1              	|  0.13146        	|  37.082         	|
|  Fit Lambda                	|  0.12437        	|  1              	|  0.12437        	|  35.08          	|
|  Fit Covariance            	|  0.077605       	|  1              	|  0.077605       	|  21.89          	|
|  Perform Eigendecomposition	|  0.0066845      	|  1              	|  0.0066845      	|  1.8855         	|
|  Save Covariance           	|  0.0056978      	|  1              	|  0.0056978      	|  1.6072         	|
|  Save Eigendecomposition   	|  0.0047404      	|  1              	|  0.0047404      	|  1.3371         	|
|  Load Covariance           	|  0.0012774      	|  1              	|  0.0012774      	|  0.36031        	|
|  Save Pairwise Score       	|  0.00080004     	|  1              	|  0.00080004     	|  0.22567        	|
|  Save Lambda               	|  0.00074937     	|  1              	|  0.00074937     	|  0.21138        	|
|  Load All Factors          	|  0.00066267     	|  1              	|  0.00066267     	|  0.18692        	|
|  Load Eigendecomposition   	|  0.00047504     	|  1              	|  0.00047504     	|  0.13399        	|
----------------------------------------------------------------------------------------------------------------------------------
```

## Counterfactual Evaluation

To run the subset removal counterfactual evaluation, please refer to the `tutorial.ipynb` notebook.
Note that `TracIn` uses the final checkpoint instead of the intermediate checkpoints throughout training.

<p align="center">
<a href="#"><img width="380" img src="figure/counterfactual.png" alt="Counterfactual"/></a>
</p>
