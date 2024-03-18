import numpy as np
import pytest
from utils.save import verify_models_equivalence

from kronfluence.utils.dataset import (
    DistributedEvalSampler,
    DistributedSamplerWithStack,
    make_indices_partition,
)
from tests.utils import prepare_test


def test_verify_models_equivalence() -> None:
    model1, _, _, _, _ = prepare_test(
        test_name="mlp",
        train_size=10,
        seed=0,
    )
    model2, _, _, _, _ = prepare_test(
        test_name="mlp",
        train_size=10,
        seed=1,
    )
    model3, _, _, _, _ = prepare_test(
        test_name="conv",
        train_size=10,
        seed=1,
    )
    assert verify_models_equivalence(model1.state_dict(), model1.state_dict())
    assert not verify_models_equivalence(model1.state_dict(), model2.state_dict())
    assert not verify_models_equivalence(model1.state_dict(), model3.state_dict())
