# Contributing

## Setup

```bash
uv sync --extra dev --extra train
```

`uv.lock` is committed. After changing dependencies in `pyproject.toml`, run
`uv lock` and commit the result, or CI's `uv sync --frozen` will fail.

## Before opening a pull request

```bash
uv run ruff check .
uv run pytest
```

Both run in CI, along with a wheel build that checks the package is importable.

## Things that are easy to get wrong here

**Landmark normalization is a contract between training and serving.**
`src/backend/detection/landmarks.py` is the only definition. Both the training
pipeline and the inference path go through it. Changing the transform
invalidates every trained checkpoint, so a change there means retraining.
`tests/test_landmarks.py` pins the behaviour, including a parity check between
the training and serving consumers that fails if a second copy ever reappears.

The transform returns **float64**, deliberately: in float32 the invariance
residual is around 4e-6, which is too large to compare two implementations for
equality. Use `to_model_input` where torch needs float32 — passing float64 to a
float32 layer raises "expected scalar type Float but found Double".

**Predictors take raw landmarks.** `StaticSignPredictor.predict` and
`DynamicSignPredictor.add_frame` normalize internally. Do not normalize before
calling them.

**Never augment before splitting.** Augmentation multiplies each sample ten
times. Splitting afterwards puts near-duplicates of the same sample in train and
test, and the reported accuracy stops meaning anything. Both training scripts
split first and augment only the training fold.

**The test split is scored once.** Early stopping and checkpoint selection use
the validation split. Reporting the number you selected on is not a
generalization estimate.

**Landmark chirality matters.** Data collection extracts from the unmirrored
frame. Mirroring only the preview is fine; mirroring before extraction records
the opposite hand and wrist-centering will not undo it.

## Linting scope

`ruff` is configured to the bug-catching subset (`E`, `F`, `B`) so CI is green on
the existing code. The cosmetic families (`I`, `UP`, `W`, `NPY002`) are listed in
`pyproject.toml` with counts. Enabling them is welcome as a standalone
mechanical commit, not mixed into a behaviour change. Do not auto-fix `NPY002`:
switching to `np.random.Generator` changes the RNG stream and therefore the
augmented training data.
