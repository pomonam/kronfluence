# WikiText & GPT-2 Example

This repository contains scripts for fine-tuning GPT-2 and computing influence scores on the WikiText2 dataset. The pipeline is inspired by [this HuggingFace example](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling).
Install the necessary packages:

```bash
pip install -r requirements.txt
```

## Training

To fine-tune GPT-2, run:

```bash
python train.py --checkpoint_dir ./checkpoints \
    --train_batch_size 8 \
    --eval_batch_size 16 \
    --learning_rate 3e-05 \
    --weight_decay 0.01 \
    --num_train_epochs 3 \
    --seed 1004
```

This will fine-tune the model using the specified hyperparameters and save the final checkpoint in the `./checkpoints` directory.

## Computing Pairwise Influence Scores

To compute pairwise influence scores using the `ekfac` factorization strategy, run the following command:

```bash
python analyze.py --query_batch_size 32 \
    --train_batch_size 64 \
    --checkpoint_dir ./checkpoints \
    --factor_strategy ekfac
```

You can also use `identity`, `diagonal`, and `kfac` for `factor_strategy`. On an A100 (80GB) GPU, this process takes approximately 50 minutes.

```
----------------------------------------------------------------------------------------------------------------------------------
|  Action                       |  Mean duration (s)    |  Num calls            |  Total time (s)       |  Percentage %         |
----------------------------------------------------------------------------------------------------------------------------------
|  Total                        |  -                    |  11                   |  2790.6               |  100 %                |
----------------------------------------------------------------------------------------------------------------------------------
|  Compute Pairwise Score       |  2253.1               |  1                    |  2253.1               |  80.739               |
|  Fit Lambda                   |  292.74               |  1                    |  292.74               |  10.49                |
|  Fit Covariance               |  194.38               |  1                    |  194.38               |  6.9654               |
|  Perform Eigendecomposition   |  22.295               |  1                    |  22.295               |  0.79893              |
|  Save Covariance              |  12.157               |  1                    |  12.157               |  0.43564              |
|  Save Eigendecomposition      |  11.641               |  1                    |  11.641               |  0.41716              |
|  Save Lambda                  |  3.0458               |  1                    |  3.0458               |  0.10915              |
|  Load Covariance              |  0.48773              |  1                    |  0.48773              |  0.017478             |
|  Load Eigendecomposition      |  0.45834              |  1                    |  0.45834              |  0.016425             |
|  Load All Factors             |  0.18377              |  1                    |  0.18377              |  0.0065855            |
|  Save Pairwise Score          |  0.10407              |  1                    |  0.10407              |  0.0037292            |
----------------------------------------------------------------------------------------------------------------------------------"
```

For more efficient computation, use half precision:

```bash
python analyze.py --query_batch_size 32 \
    --train_batch_size 64 \
    --checkpoint_dir ./checkpoints \
    --factor_strategy ekfac \
    --use_half_precision
```

This reduces computation time to about 20 minutes on an A100 (80GB) GPU:

```
----------------------------------------------------------------------------------------------------------------------------------
|  Action                       |  Mean duration (s)    |  Num calls            |  Total time (s)       |  Percentage %         |
----------------------------------------------------------------------------------------------------------------------------------
|  Total                        |  -                    |  11                   |  1211.8               |  100 %                |
----------------------------------------------------------------------------------------------------------------------------------
|  Compute Pairwise Score       |  1034.5               |  1                    |  1034.5               |  85.368               |
|  Fit Lambda                   |  88.231               |  1                    |  88.231               |  7.2811               |
|  Fit Covariance               |  59.746               |  1                    |  59.746               |  4.9305               |
|  Perform Eigendecomposition   |  14.831               |  1                    |  14.831               |  1.2239               |
|  Save Covariance              |  5.8912               |  1                    |  5.8912               |  0.48617              |
|  Save Eigendecomposition      |  5.7726               |  1                    |  5.7726               |  0.47638              |
|  Save Lambda                  |  1.624                |  1                    |  1.624                |  0.13402              |
|  Load Covariance              |  0.34494              |  1                    |  0.34494              |  0.028465             |
|  Load Eigendecomposition      |  0.33595              |  1                    |  0.33595              |  0.027724             |
|  Load All Factors             |  0.26719              |  1                    |  0.26719              |  0.022049             |
|  Save Pairwise Score          |  0.26006              |  1                    |  0.26006              |  0.021461             |
----------------------------------------------------------------------------------------------------------------------------------
```

The `half_precision_analysis.py` script compares the correlations between `float32` and `bfloat16` scores.

<p align="center">
<a href="#"><img width="380" img src="figure/half_precision.png" alt="Query Batching"/></a>
</p>

The average correlation for 481 data points is `0.96`. Finally, we can try using `torch.compile`:

```bash
python analyze.py --query_batch_size 32 \
    --train_batch_size 64 \
    --checkpoint_dir ./checkpoints \
    --factor_strategy ekfac \
    --use_half_precision \
    --use_compile
```

This reduces computation time to about 16 minutes on an A100 (80GB) GPU:

```
----------------------------------------------------------------------------------------------------------------------------------
|  Action                       |  Mean duration (s)    |  Num calls            |  Total time (s)       |  Percentage %         |
----------------------------------------------------------------------------------------------------------------------------------
|  Total                        |  -                    |  11                   |  939.4                |  100 %                |
----------------------------------------------------------------------------------------------------------------------------------
|  Compute Pairwise Score       |  735.99               |  1                    |  735.99               |  78.347               |
|  Fit Covariance               |  103.6                |  1                    |  103.6                |  11.029               |
|  Fit Lambda                   |  69.442               |  1                    |  69.442               |  7.3922               |
|  Perform Eigendecomposition   |  16.011               |  1                    |  16.011               |  1.7044               |
|  Save Covariance              |  5.9458               |  1                    |  5.9458               |  0.63294              |
|  Save Eigendecomposition      |  5.9252               |  1                    |  5.9252               |  0.63074              |
|  Save Lambda                  |  1.5185               |  1                    |  1.5185               |  0.16164              |
|  Load Covariance              |  0.42047              |  1                    |  0.42047              |  0.04476              |
|  Load Eigendecomposition      |  0.32199              |  1                    |  0.32199              |  0.034276             |
|  Load All Factors             |  0.16436              |  1                    |  0.16436              |  0.017496             |
|  Save Pairwise Score          |  0.055834             |  1                    |  0.055834             |  0.0059436            |
----------------------------------------------------------------------------------------------------------------------------------
```

## Counterfactual Experiment

`run_counterfactual.py` demonstrates a counterfactual experiment by observing the increase in validation perplexity when removing top influential sequences. 
(Note: This requires pre-computed pairwise influence scores with `ekfac` and `identity` strategies.)

<p align="center">
<a href="#"><img width="380" img src="figure/counterfactual.png" alt="Counterfactual"/></a>
</p>

## Evaluating Linear Datamodeling Score

The `evaluate_lds.py` script computes the [linear datamodeling score (LDS)](https://arxiv.org/abs/2303.14186). It measures the LDS obtained by 
retraining the network 500 times with different subsets of the dataset (5 repeats and 100 masks). We obtain `0.43` LDS (we get `0.41` LDS with the half precision).

The script also includes functionality to print out top influential sequences for a given query.

```
Query Example:
 = Homarus gammarus = 
 Homarus gammarus, known as the European lobster or common lobster, is a species of clawed lobster from the eastern Atlantic Ocean, Mediterranean Sea and parts of the Black Sea. It is closely related to the American lobster, H. americanus. It may grow to a length of 60 cm ( 24 in ) and a mass of 6 kilograms ( 13 lb ), and bears a conspicuous pair of claws. In life, the lobsters are blue, only becoming " lobster red " on cooking. Mating occurs in the summer, producing eggs which are carried by the females for up to a year before hatching into planktonic larvae. Homarus gammarus is a highly esteemed food, and is widely caught using lobster pots, mostly around the British Isles. 
 = = Description = = 
 Homarus gammarus is a large crustacean, with a body length up to 60 centimetres ( 24 in ) and weighing up to 5 – 6 kilograms ( 11 – 13 lb ), although the lobsters caught in lobster pots are usually 23 – 38 cm ( 9 – 15 in ) long and weigh 0 @.@ 7 – 2 @.@ 2 kg ( 1 @.@ 5 – 4 @.@ 9 lb ). Like other crustaceans, lobsters have a hard exoskeleton which they must shed in order to grow, in a process called ecdysis ( moulting ). This may occur several times a year for young lobsters, but decreases to once every 1 – 2 years for larger animals. 
 The first pair of pereiopods is armed with a large, asymmetrical pair of claws. The larger one is the " crusher ", and has rounded nodules used for crushing prey ; the other is the " cutter ", which has sharp inner edges, and is used for holding or tearing the prey. Usually, the left claw is the crusher, and the right is the cutter. 
 The exoskeleton is generally blue above, with spots that coalesce, and yellow below. The red colour associated with lobsters only appears after cooking. This occurs because, in life, the red pigment astaxanthin is bound to a protein complex, but the complex is broken up by the heat of cooking, releasing the red pigment. 
 The closest relative of H. gammarus is the American lobster, Homarus americanus. The two species are very similar, and can be crossed artificially

Top Influential Example:
 Sector Headquarters, Port Moresby 
 = Cape lobster = 
 The Cape lobster, Homarinus capensis, is a species of small lobster that lives off the coast of South Africa, from Dassen Island to Haga Haga. Only a few dozen specimens are known, mostly regurgitated by reef @-@ dwelling fish. It lives in rocky reefs, and is thought to lay large eggs that have a short larval phase, or that hatch directly as a juvenile. The species grows to a total length of 10 cm ( 3 @.@ 9 in ), and resembles a small European or American lobster ; it was previously included in the same genus, Homarus, although it is not very closely related to those species, and is now considered to form a separate, monotypic genus – Homarinus. Its closest relatives are the genera Thymops and Thymopides. 
 = = Distribution and ecology = = 
 The Cape lobster is endemic to South Africa. It occurs from Dassen Island, Western Cape in the west to Haga Haga, Eastern Cape in the east, a range of 900 kilometres ( 560 mi ). Most of the known specimens were regurgitated by fish caught on reefs at depths of 20 – 40 metres ( 66 – 131 ft ). This suggests that the Cape lobster inhabits rocky substrates, and may explain its apparent rarity, since such areas are not amenable to dredging or trawling, and the species may be too small to be retained by lobster traps. 
 = = Description = = 
 Homarinus capensis is considerably smaller than the large northern lobsters of the Atlantic Ocean, Homarus gammarus ( the European lobster ) and Homarus americanus ( the American lobster ), at 8 – 10 centimetres ( 3 @.@ 1 – 3 @.@ 9 in ) total length, or 4 – 5 cm ( 1 @.@ 6 – 2 @.@ 0 in ) carapace length. Accounts of the colouration of H. capensis are very variable, from tawny, red or yellow to " a rather dark olive ", similar to Homarus gammarus. 
 Homarinus and Homarus are considered to be the most plesiomorphic genera in the family Nephropidae. Nonetheless, the Cape lobster differs from Homarus in a number of characters. The rostrum of the Cape lobster is flattened, while that of Homarus is rounded in section
```