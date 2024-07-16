# OpenWebText & Llama-3-8B Example

This directory contains scripts for computing influence scores on the subset of OpenWebText dataset. 
The pipeline is inspired by the [LoggIX repository](https://github.com/logix-project/logix/tree/main/examples/language_modeling) and serves as an example of how to use Kronfluence on large-scale models. 
Install the necessary packages:

```bash
pip install -r requirements.txt
```

We will use the pre-trained Meta-Llama-3-8B model [from HuggingFace](https://huggingface.co/meta-llama/Meta-Llama-3-8B).

## Fitting EKFAC Factors

To compute factors using the `ekfac` strategy, run the following command (e.g., 4 A100 80GB GPUs):

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=4 fit_factors.py \
    --factors_name jul_13_2024 \
    --factor_batch_size 4
```

You can visualize the fitted factors using the `inspect_factors.py` script.

<p align="center">
<a href="#"><img width="380" img src="figure/eigenvalue.png" alt="Eigenvalues"/></a>
</p>

## Computing Influence Scores

The `generate.py` script contains code to generate responses from the Llama-3-8B model given certain prompts.
Some prompt-completion pairs are saved in `data/data.json`. 
To compute influence scores on the generated prompt-completion pairs using the fitted factors, run:

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=4 compute_scores.py \
    --factors_name jul_13_2024 \
    --scores_name raw \
    --train_batch_size 8 \
    --query_gradient_rank 64
```

You can experiment with various configurations. For example, after identifying the top 100 influential sequences using query batching, you can use the full query gradient (or without half precision) to re-compute influence scores on these queries. This approach can yield a more accurate ranking. (We followed this approach for the EKFAC IF paper, and it can be effective in removing outliers.)

The `/files` directory contains some results without any refinement process. Examples are available in this folder.

```
Query Sequence:
Prompt: Inflation is typically measured by; Completion:  the Consumer Price Index (CPI).

Top Influential Sequences:
================================================================================
Rank = 0; Score = 3899392.0
<|begin_of_text|>WASHINGTON (Reuters) - Both President Barack Obama and Republicans in the U.S. House of Representatives have made opening offers in negotiations to resolve the “fiscal cliff.

Here is a look at the two proposals, which are aimed at averting a more drastic combination of tax increases and spending cuts that economists say could cause a recession:

REPUBLICAN OFFER

House Republican leaders on Monday called for $2.2 trillion in new deficit reductions over 10 years.

When counting deficit reductions enacted last year, anticipated savings from winding down the wars in Iraq and Afghanistan and some interest savings, the package would amount to $4.6 trillion in reductions over a decade, according to House Republicans.

The offer made the following proposals to achieve $2.2 trillion in new deficit reductions over 10 years:

* $800 billion in new revenue through tax reform;

* unspecified healthcare program savings of $600 billion;

* other savings from changes to unspecified mandatory spending programs of $300 billion;

* tying cost-of-living increases for federal benefit programs to the Consumer Price Index to get savings of $200 billion;

* and further unspecified savings to domestic spending programs of $300 billion.

House Speaker John Boehner of Ohio and six other House Republican leaders made the offer on Monday in a letter to Obama.

DEMOCRATIC OFFER

The White House on Thursday proposed raising tax revenues by nearly $1.6 trillion, in line with what Obama has said is needed for long-term deficit reduction of nearly $4.4 trillion over 10 years.

The administration also sought $200 billion in economic stimulus from a combination of investments including infrastructure spending, extension of a payroll tax cut and jobless benefits.

The White House would also continue individual income tax cuts from the administration of former Republican President George W. Bush for all but the wealthiest earners.

Obama’s negotiators also sought the ability to raise the nation’s borrowing limit unilaterally. At present, Congress must approve an increase in the debt ceiling.

The administration’s proposal would delay across-the-board spending cuts for a year. In exchange the administration agreed to make $600 billion in spending cuts to entitlement programs.<|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|>
================================================================================
Rank = 1; Score = 3309568.0
<|begin_of_text|>India's inflation rate has slumped to the lowest level of modern times and there is much talk that the Reserve Bank of India will lower interest rates to stimulate the economy--that might take a little longer than many are predicting. We must always distinguish between general inflation, a general upward pressure on the price level, and specific price changes. This current low inflation rate looks much more like the effect of the latter, thus interest rate changes aren't so warranted.

Growth in industrial production fell to a three-month low in May while consumer price index (CPI)-based inflation declined below a stipulated floor of 2 per cent in June, providing the Reserve Bank of India leeway to cut the policy interest rate in August.

The RBI certainly has the power and the room to cut rates, yes, but that's not quite the correct policy decision here. Which is whether the RBI should cut interest rates as a result of the low inflation and that's much less certain:

Data released by the Central Statistics Office (CSO) on Wednesday showed retail inflation, as measured by the consumer price index (CPI), rose an annual 1.5% in June, slower than previous month's 2.2%. This was the slowest pace of increase since the government unveiled the new retail inflation series in 2012. The previous lows were in 1999 and in 1978 under a different series.

My point being that CPI isn't really the correct inflation measure to be using when considering interest rate changes and inflation. The US Federal Reserve, for example, uses the PCE rate although that's not really the important point here. CPI, PCE, RPI, they all have their slight differences and are useful in slightly different manners. But there's a much more important difference we need to take account of:

Aside from the steep fall in food inflation, which has been downplayed by the RBI time and again, the steady and consistent fall in core inflation (non-food and fuel) could find favour with the central bank.

It's this difference between core and non-core inflation that matters. It does depend upon which government and which statistical system we're talking about as to whether all produce core and non-core PCE and RPI and CPI, but in general the distinction is understood between the two different types of inflation rate:

Data on Wednesday showed headline consumer price inflation fell to 1.5 percent in the year to June from an annual 2.2 percent a month ago and below forecasts for a 1.6 percent reading
================================================================================
Rank = 2; Score = 3244032.0
<|begin_of_text|>Inflation accounting comprises a range of accounting models designed to correct problems arising from historical cost accounting in the presence of high inflation and hyperinflation.[1] [2] For example, in countries experiencing hyperinflation the International Accounting Standards Board requires corporations to implement financial capital maintenance in units of constant purchasing power in terms of the monthly published Consumer Price Index. This does not result in capital maintenance in units of constant purchasing power since that can only be achieved in terms of a daily index.

Historical cost basis in financial statements [ edit ]

Fair value accounting (also called replacement cost accounting or current cost accounting) was widely used in the 19th and early 20th centuries, but historical cost accounting became more widespread after values overstated during the 1920s were reversed during the Great Depression of the 1930s. Most principles of historical cost accounting were developed after the Wall Street Crash of 1929, including the presumption of a stable currency.[3]

Measuring unit principle [ edit ]

Under a historical cost-based system of accounting, inflation leads to two basic problems. First, many of the historical numbers appearing on financial statements are not economically relevant because prices have changed since they were incurred. Second, since the numbers on financial statements represent dollars expended at different points of time and, in turn, embody different amounts of purchasing power, they are simply not additive. Hence, adding cash of $10,000 held on December 31, 2002, with $10,000 representing the cost of land acquired in 1955 (when the price level was significantly lower) is a dubious operation because of the significantly different amount of purchasing power represented by the two numbers.[4]

By adding dollar amounts that represent different amounts of purchasing power, the resulting sum is misleading, as one would be adding 10,000 dollars to 10,000 Euros to get a total of 20,000. Likewise subtracting dollar amounts that represent different amounts of purchasing power may result in an apparent capital gain which is actually a capital loss. If a building purchased in 1970 for $20,000 is sold in 2006 for $200,000 when its replacement cost is $300,000, the apparent gain of $180,000 is illusory.

Misleading reporting under historical cost accounting [ edit ]

"In most countries, primary financial statements are prepared on the historical cost basis of accounting without regard either to changes in the general level of prices or to increases in specific prices of assets held, except to the extent that property, plant
================================================================================
Rank = 3; Score = 3096576.0
<|begin_of_text|>A Wall St. sign is seen outside the entrance of NYSE Thomson Reuters By Tanya Agrawal

(Reuters) - U.S. stock index futures fell on Wednesday as Chinese stocks had another roller coaster ride and as investors await the minutes from last month's Federal Reserve meeting for clues on when interest rates will be increased.

* While the health of the U.S. economy appears to be stabilizing, the effect of the yuan devaluation and other macro factors are playing on investor's minds. The Fed minutes will be released at 2 p.m. ET.

* Economists believe the Fed will probably raise rates twice this year, with the first hike coming in September. Investors are still not fully convinced of a September hike, but most are betting a rate hike will occur by the end of year.

* Chinese stocks reversed sharp declines and ended higher after the central bank injected more funds into the financial system for a second day in a bid to calm panicky markets.

* The People's Bank of China devalued the yuan last week, triggering an avalanche of selling by investors globally who feared Beijing wanted to engineer a much sharper decline to support weak exports.

* The Chinese market gyrations kept commodity prices under pressure, with oil and copper near six-year lows.

* Other data due Wednesday is expected to show consumer prices rose 0.2 percent in July, less than the 0.3 percent rise in June. The Consumer Price Index data is due at 8:30 a.m. ET.

* Lowe's shares fell 1.6 percent to $71.85 in premarket trading after the No.2 U.S. home improvement chain's quarterly profit missed expectations.

* Yum Brands rose 1.5 percent to $85.50, a day after the owner of the KFC and Pizza Hut chains announced new leadership for its China division as activist investors lobby the company to spin off that business.

* Staples fell 1.6 percent to $13.93 after the office supplies retailer reported quarterly revenue slightly below analysts' estimates, hurt by a stronger dollar.

Futures snapshot at 7:07 a.m. ET:

* S&P 500 e-minis were down 5.25 points, or 0.25 percent, with 143,877 contracts traded.

* Nasdaq 100 e-minis were down 11 points, or 0.24 percent, on volume of 23,476 contracts.

* Dow e-minis <1YMc1> were down 57 points,
```
