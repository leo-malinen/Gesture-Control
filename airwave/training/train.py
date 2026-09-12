"""Training entry point for the custom-gesture classifier (M5).

Thin by design: the interesting code is in ``vision/model.py``. What lives
here is the reporting, because a training script that prints one accuracy
number is a script you cannot debug. Per-class counts catch the "I recorded
four samples of spock" problem, and the confusion matrix tells you *which*
two gestures the model cannot separate - almost always the answer is that
they genuinely look alike and one of them needs re-recording.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..vision.model import NumpyMLP, train_mlp
from .recorder import DEFAULT_DATA_DIR, load_dataset

log = logging.getLogger("airwave.training.train")

MIN_SAMPLES_PER_CLASS = 15


def train_from_dir(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    out_path: str | Path = "data/model.npz",
    *,
    epochs: int = 400,
    hidden: tuple[int, ...] = (64, 32),
    seed: int = 0,
    verbose: bool = True,
) -> tuple[NumpyMLP, dict]:
    X, y, counts = load_dataset(data_dir)
    thin = {label: n for label, n in counts.items() if n < MIN_SAMPLES_PER_CLASS}
    if thin:
        log.warning(
            "these gestures have very few samples and will be unreliable: %s (aim for %d+ each)",
            ", ".join(f"{k}={v}" for k, v in thin.items()),
            MIN_SAMPLES_PER_CLASS,
        )
    model, metrics = train_mlp(X, y, hidden=hidden, epochs=epochs, seed=seed, verbose=verbose)
    model.save(out_path)
    metrics["out_path"] = str(out_path)
    metrics["counts"] = counts
    return model, metrics


def format_report(model: NumpyMLP, metrics: dict) -> str:
    lines = [
        "",
        f"trained on {metrics['n_samples']} samples across {len(model.labels)} gestures",
        f"  train accuracy  {metrics['train_accuracy']:.3f}",
        f"  val accuracy    {metrics['val_accuracy']:.3f}",
        f"  saved to        {metrics.get('out_path')}",
        "",
        "samples per gesture:",
    ]
    for label, count in sorted(metrics.get("counts", {}).items()):
        lines.append(f"  {label:<14} {count}")

    confusion = metrics.get("confusion")
    if confusion:
        matrix = np.array(confusion)
        width = max(len(l) for l in model.labels) + 1
        lines += ["", "validation confusion (rows = truth, cols = predicted):",
                  " " * (width + 1) + " ".join(f"{l[:4]:>4}" for l in model.labels)]
        for label, row in zip(model.labels, matrix):
            lines.append(f"  {label:<{width}}" + " ".join(f"{v:>4}" for v in row))
        confusable = [
            (model.labels[i], model.labels[j], int(matrix[i, j]))
            for i in range(len(model.labels))
            for j in range(len(model.labels))
            if i != j and matrix[i, j] > 0
        ]
        if confusable:
            lines += ["", "most confused pairs:"]
            for truth, pred, n in sorted(confusable, key=lambda t: -t[2])[:5]:
                lines.append(f"  {truth} read as {pred} ({n}x) - record more {truth}, or pick a more distinct pose")
    lines.append("")
    lines.append("switch to it with:  settings.classifier: model   in airwave.yaml")
    return "\n".join(lines)
