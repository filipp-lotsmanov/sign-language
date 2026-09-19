"""
Regression tests for split ordering.

The bug these exist to prevent: train.py augmented x10 and split afterwards,
so ten near-copies of every source sample were spread across train, val and
test. The reported 99.8% measured recall of images the model had trained on.

Note what is NOT tested here, deliberately: that augmented copies are absent
from the held-out splits. That cannot be checked by distance - a copy lands a
median 2.9 from its source while distinct sources sit 9.9 apart, and mirror_x
throws the tail past 15. It is guaranteed structurally instead, by splitting
before augmenting, which is what test_augmenting_train_leaves_heldout_untouched
pins down.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

TRAINING_STATIC = Path(__file__).resolve().parent.parent / "training" / "static"


def _load(module_name: str):
    """Load a training script by path; training/ is not an installed package."""
    sys.path.insert(0, str(TRAINING_STATIC))
    try:
        spec = importlib.util.spec_from_file_location(
            f"_training_{module_name}", TRAINING_STATIC / f"{module_name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(TRAINING_STATIC))


dataset_creation = _load("dataset_creation")
data_augmentation = _load("data_augmentation")


@pytest.fixture
def samples():
    """120 samples across 4 classes, each a plausible hand."""
    rng = np.random.default_rng(0)
    out = []
    for letter in "ABCD":
        for row in rng.uniform(-1, 1, (30, 63)) * 0.15:
            row = row.copy()
            row[0:3] = 0.0
            row[27:30] = [0.0, -0.08, 0.0]
            out.append((letter, row))
    return out


class TestSplitBeforeAugment:
    def test_splits_partition_the_input(self, samples) -> None:
        train, val, test = dataset_creation.split_samples(samples, 0.15, 0.15, seed=42)
        assert len(train) + len(val) + len(test) == len(samples)

    def test_no_sample_appears_in_two_splits(self, samples) -> None:
        train, val, test = dataset_creation.split_samples(samples, 0.15, 0.15, seed=42)
        ids = [{id(coords) for _, coords in split} for split in (train, val, test)]
        assert ids[0].isdisjoint(ids[1])
        assert ids[0].isdisjoint(ids[2])
        assert ids[1].isdisjoint(ids[2])

    def test_every_class_present_in_every_split(self, samples) -> None:
        train, val, test = dataset_creation.split_samples(samples, 0.15, 0.15, seed=42)
        for split in (train, val, test):
            assert {label for label, _ in split} == set("ABCD")

    def test_split_is_deterministic(self, samples) -> None:
        a = dataset_creation.split_samples(samples, 0.15, 0.15, seed=42)[2]
        b = dataset_creation.split_samples(samples, 0.15, 0.15, seed=42)[2]
        np.testing.assert_array_equal([c for _, c in a], [c for _, c in b])

    def test_augmenting_train_leaves_heldout_untouched(self, samples) -> None:
        """The structural guarantee: augmentation only ever sees train."""
        train, val, test = dataset_creation.split_samples(samples, 0.15, 0.15, seed=42)
        val_before = [coords.copy() for _, coords in val]
        test_before = [coords.copy() for _, coords in test]

        grown = data_augmentation.augment_by_class(train, 10)

        assert len(grown) > len(train)
        assert len(val) == len(val_before) and len(test) == len(test_before)
        for (_, after), before in zip(val, val_before, strict=True):
            np.testing.assert_array_equal(after, before)
        for (_, after), before in zip(test, test_before, strict=True):
            np.testing.assert_array_equal(after, before)

    def test_tiny_class_is_rejected_not_silently_dropped(self, samples) -> None:
        samples = samples + [("Z", np.zeros(63))]
        with pytest.raises(ValueError, match="fewer than 3 samples"):
            dataset_creation.split_samples(samples, 0.15, 0.15, seed=42)


class TestDuplicateGuard:
    def test_passes_on_disjoint_data(self) -> None:
        rng = np.random.default_rng(0)
        dataset_creation.assert_no_duplicate_rows_across_splits(
            rng.normal(size=(50, 63)), rng.normal(size=(10, 63)), "test"
        )

    def test_fires_on_an_exact_duplicate(self) -> None:
        rng = np.random.default_rng(0)
        train = rng.normal(size=(50, 63))
        held_out = np.vstack([rng.normal(size=(9, 63)), train[3]])
        with pytest.raises(RuntimeError, match="DUPLICATE ROWS"):
            dataset_creation.assert_no_duplicate_rows_across_splits(train, held_out, "test")

    def test_empty_inputs_are_a_no_op(self) -> None:
        dataset_creation.assert_no_duplicate_rows_across_splits(
            np.zeros((0, 63)), np.zeros((5, 63)), "test"
        )


class TestEncodeSplits:
    def test_label_indices_match_across_splits(self, samples) -> None:
        train, val, test = dataset_creation.split_samples(samples, 0.15, 0.15, seed=42)
        frames = [
            dataset_creation.samples_to_dataframe(dataset_creation.normalize_samples(s))
            for s in (train, val, test)
        ]
        X_train, y_train, X_val, y_val, X_test, y_test, encoder = dataset_creation.encode_splits(
            *frames
        )

        assert list(encoder.classes_) == list("ABCD")
        assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1] == 63
        for y in (y_train, y_val, y_test):
            assert set(y) <= set(range(len(encoder.classes_)))
        assert len(y_train) == len(X_train)
