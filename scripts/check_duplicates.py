"""
Detect augmented and duplicated rows in a landmark CSV.

The question this answers: was this CSV already augmented before it was saved?
It matters because reordering train.py only fixes the augmentation THIS
pipeline applies (to the LETTER_FOLDERS classes). If ngt_data.csv arrived
pre-augmented from an earlier dataset_builder run, its copies are still spread
across every split and reordering train.py does not help.

How it works, and why not the obvious way: clustering near-duplicates does NOT
recover the original sample count. Measured on this project's augmentation, a
copy sits a median 2.9 from its source while distinct sources sit 9.9 apart,
but mirror_x throws the tail past 15, so sibling groups overlap their
neighbours and no radius gives a stable cluster count.

What does work is comparing classes against each other. Augmented rows are
packed much more tightly than genuine ones, so a class whose median
nearest-neighbour distance is far below the cross-class median is augmented.
On a synthetic set where four of eight classes were augmented x10, augmented
classes scored 2.3 against 9.0 for the rest and all four were identified with
no false positives.

Usage:
    python scripts/check_duplicates.py ngt_final.csv
    python scripts/check_duplicates.py ngt_data.csv --threshold 0.6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# A class is flagged when its median nearest-neighbour distance falls below
# this fraction of the cross-class median. 0.6 sits well clear of both sides
# of the separation measured above (augmented ~0.41, genuine ~1.58).
DEFAULT_THRESHOLD = 0.6


def load(csv_path: Path):
    if not csv_path.exists():
        sys.exit(f"Not found: {csv_path}")
    frame = pd.read_csv(csv_path)
    if "label" not in frame.columns:
        sys.exit(f"{csv_path} has no 'label' column; columns: {list(frame.columns)[:6]}")

    drop = [c for c in ("label", "split") if c in frame.columns]
    features = frame.drop(columns=drop).select_dtypes("number").values.astype(np.float64)
    print(
        f"{csv_path}: {len(frame):,} rows, {features.shape[1]} feature columns, "
        f"{frame['label'].nunique()} classes"
    )
    if "split" in frame.columns:
        print(f"  split column: {dict(frame['split'].value_counts())}")
    return frame, features


def nn_distances(X: np.ndarray) -> np.ndarray:
    """Distance from each row to the closest other row in the same class."""
    if len(X) < 2:
        return np.array([])
    distances, _ = NearestNeighbors(n_neighbors=2).fit(X).kneighbors(X)
    return distances[:, 1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("csv", type=Path, help="ngt_data.csv or ngt_final.csv")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"flag classes below this ratio (default {DEFAULT_THRESHOLD})",
    )
    args = parser.parse_args()

    frame, X = load(args.csv)
    labels = frame["label"].values

    exact = len(frame) - len(np.unique(X, axis=0))
    print(
        f"  exact duplicate rows: {exact:,}"
        + ("   <- byte-identical rows, deduplicate before splitting" if exact else "")
    )

    stats = {}
    for label in sorted(set(labels)):
        distances = nn_distances(X[labels == label])
        if len(distances):
            stats[label] = (int((labels == label).sum()), float(np.median(distances)))

    if len(stats) < 2:
        sys.exit("Need at least two classes with two rows each to compare.")

    baseline = float(np.median([median for _, median in stats.values()]))
    print(f"\ncross-class median of per-class median NN distance: {baseline:.4f}")
    print("\n(ratio below 1 means the class is packed tighter than its peers)")
    print(f"{'class':<12}{'rows':>9}{'median NN':>13}{'ratio':>9}")

    flagged = []
    for label, (count, median) in sorted(stats.items(), key=lambda kv: kv[1][1]):
        ratio = median / baseline if baseline else float("nan")
        mark = ""
        if ratio < args.threshold:
            mark = "  <- augmented"
            flagged.append(label)
        print(f"{label:<12}{count:>9,}{median:>13.4f}{ratio:>9.2f}{mark}")

    print()
    if flagged:
        print(f"Classes that look augmented: {', '.join(flagged)}")
        print("Their row counts are inflated, so the real per-class sample size is")
        print("much smaller than the label counts suggest. If any of these came from")
        print("the original CSV rather than LETTER_FOLDERS, then that CSV was saved")
        print("pre-augmented and reordering train.py does not fix its leakage -")
        print("those rows need regenerating from source images without augmentation.")
    else:
        print("No class stands out as augmented. If the row counts still look high,")
        print("the duplication (if any) is uniform across classes, which this")
        print("comparison cannot see - it only detects relative differences.")


if __name__ == "__main__":
    main()
