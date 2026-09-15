"""One evaluation protocol, shared by both backends.

Accuracy alone hides a lot on a 29-class problem, so every run also stores the
macro F1, the per-class report and the confusion matrix. The numbers land in
``metrics.json`` and the tables in ``artifacts/`` of the run directory.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def compute_metrics(y_true: list[str], y_pred: list[str], classes: list[str]) -> dict[str, object]:
    """Headline metrics plus the per-class breakdown."""
    report = classification_report(
        y_true, y_pred, labels=classes, output_dict=True, zero_division=0
    )
    weakest = sorted(
        ((label, report[label]["f1-score"]) for label in classes if label in report),
        key=lambda item: item[1],
    )[:5]
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "weighted_f1": round(
            float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4
        ),
        "n_samples": len(y_true),
        "n_classes": len(classes),
        "weakest_classes": [{"label": label, "f1": round(float(f1), 4)} for label, f1 in weakest],
    }


def classification_table(y_true: list[str], y_pred: list[str], classes: list[str]) -> pd.DataFrame:
    report = classification_report(
        y_true, y_pred, labels=classes, output_dict=True, zero_division=0
    )
    rows = [
        {
            "label": label,
            "precision": round(float(report[label]["precision"]), 4),
            "recall": round(float(report[label]["recall"]), 4),
            "f1": round(float(report[label]["f1-score"]), 4),
            "support": int(report[label]["support"]),
        }
        for label in classes
        if label in report
    ]
    return pd.DataFrame(rows)


def confusion_frame(y_true: list[str], y_pred: list[str], classes: list[str]) -> pd.DataFrame:
    matrix = confusion_matrix(y_true, y_pred, labels=classes)
    return pd.DataFrame(matrix, index=classes, columns=classes)


def top_confusions(matrix: pd.DataFrame, limit: int = 10) -> list[dict[str, object]]:
    """The pairs the model actually mixes up - the most useful line of the report."""
    values = matrix.to_numpy().copy()
    np.fill_diagonal(values, 0)
    flat = np.argsort(values, axis=None)[::-1][:limit]
    pairs = []
    for index in flat:
        i, j = divmod(int(index), values.shape[1])
        if values[i, j] == 0:
            continue
        pairs.append(
            {"true": matrix.index[i], "predicted": matrix.columns[j], "count": int(values[i, j])}
        )
    return pairs


def save_confusion_figure(matrix: pd.DataFrame, path: str | Path) -> Path | None:
    """Render the confusion matrix. Skipped silently when matplotlib is absent."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - optional dependency
        return None

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = matrix.div(matrix.sum(axis=1).replace(0, 1), axis=0)

    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(normalized.to_numpy(), cmap="magma", vmin=0, vmax=1)
    ax.set_title("Confusion matrix (row-normalised)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_yticks(range(len(matrix.index)))
    ax.set_xticklabels(matrix.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(matrix.index, fontsize=7)
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def evaluation_artifacts(
    y_true: list[str],
    y_pred: list[str],
    classes: list[str],
    directory: str | Path,
) -> dict[str, object]:
    """Compute everything and write the tables next to the model."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    metrics = compute_metrics(y_true, y_pred, classes)
    table = classification_table(y_true, y_pred, classes)
    matrix = confusion_frame(y_true, y_pred, classes)

    table.to_csv(directory / "classification_report.csv", index=False)
    matrix.to_csv(directory / "confusion_matrix.csv")
    save_confusion_figure(matrix, directory / "confusion_matrix.png")

    metrics["top_confusions"] = top_confusions(matrix)
    return metrics


__all__ = [
    "classification_table",
    "compute_metrics",
    "confusion_frame",
    "evaluation_artifacts",
    "save_confusion_figure",
    "top_confusions",
]
