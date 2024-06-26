# GLUE & BERT Example

This directory contains scripts for fine-tuning BERT and computing influence scores on GLUE benchmark. The pipeline is motivated from [this HuggingFace Example](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification).
To get started, please install the necessary packages:

```bash
pip install -r requirements.txt
```

## Training

To fine-tune BERT on a specific dataset, run the following command (we are using the `SST2` dataset in this example):

```bash
python train.py --dataset_name sst2 \
    --checkpoint_dir ./checkpoints \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --learning_rate 3e-05 \
    --weight_decay 0.01 \
    --num_train_epochs 3 \
    --seed 1004
```

## Computing Pairwise Influence Scores

To obtain pairwise influence scores on a maximum of 2000 query data points using `ekfac`, run the following command:

```bash
python analyze.py --dataset_name sst2 \
    --query_batch_size 175 \
    --train_batch_size 128 \
    --checkpoint_dir ./checkpoints \
    --factor_strategy ekfac
```

On an A100 (80GB), it takes roughly 95 minutes to compute the pairwise scores for SST2 with around 900 query data points (including computing EKFAC factors):

```

```

For more efficient computation, use half precision:

```bash
python analyze.py --dataset_name sst2 \
    --query_batch_size 175 \
    --train_batch_size 128 \
    --checkpoint_dir ./checkpoints \
    --factor_strategy ekfac \
    --use_half_precision
```

This reduces computation time to about 20 minutes on an A100 (80GB) GPU:

```

```

## Counterfactual Evaluation

Can we remove top positively influential training examples to make some queries misclassify? Subset removal counterfactual evaluation 
selects correctly classified query data point, removes top-k positively influential training samples, and retrain the network with the modified dataset to see if that query 
data point gets misclassified.

We first need to compute pairwise influence scores for the `RTE` dataset:

```bash
python train.py --dataset_name rte \
    --checkpoint_dir ./checkpoints \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --learning_rate 2e-05 \
    --weight_decay 0.01 \
    --num_train_epochs 3 \
    --seed 1004

python analyze.py --dataset_name rte \
    --query_batch_size 175 \
    --train_batch_size 128 \
    --checkpoint_dir ./checkpoints \
    --factor_strategy ekfac
    
python analyze.py --dataset_name rte \
    --query_batch_size 175 \
    --train_batch_size 128 \
    --checkpoint_dir ./checkpoints \
    --factor_strategy identity
```

`run_counterfactual.py` contains the script to run the counterfactual experiment.

<p align="center">
<a href="#"><img width="380" img src="figure/counterfactual.png" alt="Counterfactual"/></a>
</p>

## Evaluating Linear Datamodeling Score

The `evaluate_lds.py` script computes the [linear datamodeling score (LDS)](https://arxiv.org/abs/2303.14186). It measures the LDS obtained by 
retraining the network 500 times with different subsets of the dataset (5 repeats and 100 masks). By running `evaludate_lds.py`, we obtain `xx` LDS (we get `xx` LDS with the half precision).

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