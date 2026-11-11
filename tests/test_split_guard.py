"""Michigan datasets must never fall back to a randomly generated split."""
import pytest

from datasets.data_module import (REQUIRE_PREMADE_SPLITS, SPLIT_DIR_ALIASES,
                                  validate_premade_split)


@pytest.mark.unit
def test_full_block_shares_michigan_splits():
    assert SPLIT_DIR_ALIASES["MICHIGAN_FULL_BLOCK"] == "MICHIGAN"


@pytest.mark.unit
@pytest.mark.parametrize("name", ["MICHIGAN", "MICHIGAN_FULL_BLOCK", "MICHIGAN_GENDER",
                                  "MICHIGAN_WITHIN", "MICHIGAN_POOLED"])
def test_all_michigan_datasets_require_premade_splits(name):
    assert name in REQUIRE_PREMADE_SPLITS


@pytest.mark.unit
def test_valid_loocv_split_passes():
    validate_premade_split(["a", "b", "c"], ["d", "e"], ["f"], "split_fixed_101.txt")


@pytest.mark.unit
@pytest.mark.parametrize("tr,va,te,msg", [
    (["a"], ["b", "c"], ["b", "c"], "val/test overlap"),   # what determine_split_randomly writes
    (["a", "f"], ["b"], ["f"], "train/test overlap"),
    (["a", "b"], ["b"], ["c"], "train/val overlap"),
    (["a"], ["b"], [], "empty set"),
])
def test_invalid_splits_rejected(tr, va, te, msg):
    with pytest.raises(ValueError, match=msg):
        validate_premade_split(tr, va, te, "split_fixed_1.txt")
