"""Train/test split policies.

The ASL Alphabet images were captured as continuous bursts: within a class,
consecutive files are near-duplicate frames of the same hand, same lighting,
same session. A stratified *random* split therefore puts near-identical frames
on both sides and inflates test accuracy - the model can score very high while
having learned little that transfers to a new hand or a new webcam.

``blocked`` keeps the frame order and holds out the tail of each class, which is
a closer proxy for "unseen recording". Both policies are implemented so the gap
between them can be measured instead of assumed. See docs/evaluation.md.
"""

from __future__ import annotations

import math

import pandas as pd
from sklearn.model_selection import train_test_split

from asl.config import SplitConfig

TRAIN = "train"
TEST = "test"


def make_split(catalog: pd.DataFrame, config: SplitConfig) -> pd.DataFrame:
    """Return a copy of ``catalog`` with a ``split`` column in {train, test}."""
    if config.strategy == "random":
        return _random_split(catalog, config)
    if config.strategy == "blocked":
        return _blocked_split(catalog, config)
    raise ValueError(f"unknown split strategy: {config.strategy}")  # pragma: no cover


def _random_split(catalog: pd.DataFrame, config: SplitConfig) -> pd.DataFrame:
    out = catalog.copy()
    counts = out["label"].value_counts()
    stratify = out["label"] if counts.min() >= 2 else None
    train_idx, test_idx = train_test_split(
        out.index.to_numpy(),
        test_size=config.test_size,
        random_state=config.seed,
        shuffle=True,
        stratify=stratify,
    )
    out["split"] = TRAIN
    out.loc[test_idx, "split"] = TEST
    return out


def _blocked_split(catalog: pd.DataFrame, config: SplitConfig) -> pd.DataFrame:
    out = catalog.sort_values(["label", "order"]).copy()
    out["split"] = TRAIN
    for _label, group in out.groupby("label", sort=False):
        n_test = max(1, math.floor(len(group) * config.test_size)) if len(group) > 1 else 0
        if n_test:
            out.loc[group.index[-n_test:], "split"] = TEST
    return out.sort_index()


def split_summary(catalog: pd.DataFrame) -> dict[str, object]:
    counts = catalog["split"].value_counts()
    per_class = (
        catalog.groupby(["label", "split"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
        .to_dict("index")
    )
    return {
        "n_train": int(counts.get(TRAIN, 0)),
        "n_test": int(counts.get(TEST, 0)),
        "per_class": {
            str(label): {str(split): int(value) for split, value in row.items()}
            for label, row in per_class.items()
        },
    }


__all__ = ["TEST", "TRAIN", "make_split", "split_summary"]
