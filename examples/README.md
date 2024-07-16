# Kronfluence: Examples

For detailed technical documentation of Kronfluence, please refer to the [Technical Documentation](https://github.com/pomonam/kronfluence/blob/main/DOCUMENTATION.md) page.

## Getting Started

To run all examples, install the necessary packages:

```bash
pip install -r requirements.txt
```

Alternatively, navigate to each example folder and run `pip install -r requirements.txt`.

## List of Tasks

Our examples cover the following tasks:

<div align="center">

| Task                 |                                                                            Example Datasets	                                                                            |
|----------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| Regression           |                                                  [UCI](https://github.com/pomonam/kronfluence/tree/main/examples/uci)                                                   |
| Image Classification |      [CIFAR-10](https://github.com/pomonam/kronfluence/tree/main/examples/cifar) & [ImageNet](https://github.com/pomonam/kronfluence/tree/main/examples/imagenet)       |
| Text Classification  |                                                 [GLUE](https://github.com/pomonam/kronfluence/tree/main/examples/glue)                                                  |
| Multiple-Choice      |                                                 [SWAG](https://github.com/pomonam/kronfluence/tree/main/examples/swag)                                                  |
| Summarization        |                              [CNN/DailyMail](https://github.com/pomonam/kronfluence/tree/main/examples/dailymail)                                                       |
| Language Modeling    | [WikiText-2](https://github.com/pomonam/kronfluence/tree/main/examples/wikitext) & [OpenWebText](https://github.com/pomonam/kronfluence/tree/main/examples/openwebtext) |

</div>

These examples demonstrate various use cases of Kronfluence, including the usage of AMP (Automatic Mixed Precision) and DDP (Distributed Data Parallel). 
Many examples aim to replicate the settings used in [our paper](https://arxiv.org/abs/2405.12186). If you would like to see more examples added to this repository, please leave an issue.
