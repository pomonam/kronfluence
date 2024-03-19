# pylint: skip-file

from torch.utils.data import DataLoader
from transformers import default_data_collator

from tests.testable_tasks.classification import (
    make_classification_dataset,
    make_conv_model,
)
from tests.testable_tasks.language_modeling import make_gpt_dataset, make_tiny_gpt
from tests.testable_tasks.regression import make_mlp_model, make_regression_dataset
from tests.testable_tasks.text_classification import make_bert_dataset, make_tiny_bert


def test_mlp():
    model = make_mlp_model(bias=True, seed=0)
    dataset = make_regression_dataset(num_data=64, seed=0)
    batch_size = 4
    loader = DataLoader(
        dataset,
        collate_fn=None,
        batch_size=batch_size,
        drop_last=False,
        shuffle=False,
    )

    batch = next(iter(loader))
    output = model(batch[0])
    assert output.shape == batch[1].shape
    output.sum().backward()


def test_conv():
    model = make_conv_model(bias=True, seed=0)
    dataset = make_classification_dataset(num_data=64, seed=0)
    batch_size = 8
    loader = DataLoader(
        dataset,
        collate_fn=None,
        batch_size=batch_size,
        drop_last=False,
        shuffle=False,
    )

    batch = next(iter(loader))
    logits = model(batch[0])
    logits.sum().backward()


def test_bert():
    model = make_tiny_bert(seed=0)
    dataset = make_bert_dataset(num_data=8, seed=0)
    batch_size = 8
    loader = DataLoader(
        dataset,
        collate_fn=default_data_collator,
        batch_size=batch_size,
        drop_last=False,
        shuffle=False,
    )

    batch = next(iter(loader))
    inputs = (
        batch["input_ids"],
        batch["attention_mask"],
        batch["token_type_ids"],
    )
    logits = model(*inputs).logits
    logits.sum().backward()


def test_gpt():
    model = make_tiny_gpt(seed=0)
    dataset = make_gpt_dataset(num_data=8, seed=0)
    batch_size = 8
    loader = DataLoader(
        dataset,
        collate_fn=default_data_collator,
        batch_size=batch_size,
        drop_last=False,
        shuffle=False,
    )

    batch = next(iter(loader))
    inputs = (
        batch["input_ids"],
        None,
        batch["attention_mask"],
    )
    logits = model(*inputs).logits
    logits.sum().backward()
