"""Trained gesture classifier (M5) - a drop-in replacement for ``rules.py``.

Deliberately a small numpy MLP rather than scikit-learn or PyTorch. The
dataset this trains on is a few hundred hand-recorded samples of a dozen
classes over 82 features; a two-layer net fits it in under a second on CPU,
and writing it out in numpy keeps ``requirements.txt`` at the size the PRD's
install goal implies. The interface is the thing that matters - anyone who
wants a bigger model swaps this class and changes nothing else.

Two design choices earn their keep:

* **Standardization is stored with the weights.** A model whose normalization
  constants live somewhere else is a model that silently mispredicts the day
  someone retrains on a different dataset.
* **Low-confidence predictions become ``none``.** A softmax always returns
  something; without a floor, a hand shape the model has never seen gets
  confidently labelled as whatever class is nearest, and fires an action. The
  rules classifier can only ever be wrong between known poses - the trained one
  needs an explicit "I don't know".
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..gestures import NONE_LABEL
from .classifier import Classification
from .normalize import FEATURE_DIM, feature_vector

log = logging.getLogger("airwave.vision.model")


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


@dataclass
class NumpyMLP:
    """Two-hidden-layer ReLU network with softmax output, trained with Adam."""

    weights: list[np.ndarray]
    biases: list[np.ndarray]
    labels: list[str]
    mean: np.ndarray
    std: np.ndarray
    meta: dict[str, Any]

    # ------------------------------------------------------------- inference

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std

    def forward(self, X: np.ndarray) -> np.ndarray:
        a = self._standardize(np.atleast_2d(X).astype(np.float32))
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            a = a @ W + b
            if i < len(self.weights) - 1:
                a = np.maximum(a, 0.0)
        return _softmax(a)

    def predict(self, X: np.ndarray) -> tuple[list[str], np.ndarray]:
        probs = self.forward(X)
        idx = probs.argmax(axis=1)
        return [self.labels[i] for i in idx], probs.max(axis=1)

    # ------------------------------------------------------------ persistence

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "labels": json.dumps(self.labels),
            "meta": json.dumps(self.meta),
            "mean": self.mean,
            "std": self.std,
            "n_layers": np.array(len(self.weights)),
        }
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            payload[f"W{i}"] = W
            payload[f"b{i}"] = b
        np.savez_compressed(path, **payload)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "NumpyMLP":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"no trained model at {path}. Record samples with 'python -m airwave record <gesture>' "
                f"and train with 'python -m airwave train', or set settings.classifier: rules"
            )
        data = np.load(path, allow_pickle=False)
        n = int(data["n_layers"])
        return cls(
            weights=[data[f"W{i}"] for i in range(n)],
            biases=[data[f"b{i}"] for i in range(n)],
            labels=json.loads(str(data["labels"])),
            mean=data["mean"],
            std=data["std"],
            meta=json.loads(str(data["meta"])),
        )


def train_mlp(
    X: np.ndarray,
    y: Sequence[str],
    *,
    hidden: tuple[int, ...] = (64, 32),
    epochs: int = 400,
    batch_size: int = 32,
    lr: float = 3e-3,
    weight_decay: float = 1e-4,
    seed: int = 0,
    val_split: float = 0.2,
    verbose: bool = True,
) -> tuple[NumpyMLP, dict[str, Any]]:
    """Fit a classifier on recorded samples. Returns the model and its metrics.

    The validation split is stratified because hand-recorded datasets are
    always imbalanced - people record forty of the gesture they care about and
    eight of the neutral pose - and an unstratified split can leave a class
    entirely out of validation, producing a meaningless accuracy number.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float32)
    labels = sorted(set(y))
    if len(labels) < 2:
        raise ValueError(f"need at least 2 gesture classes to train, got {labels}")
    index = {label: i for i, label in enumerate(labels)}
    Y = np.array([index[label] for label in y], dtype=np.int64)

    train_idx, val_idx = _stratified_split(Y, val_split, rng)
    Xtr, Ytr = X[train_idx], Y[train_idx]
    Xva, Yva = X[val_idx], Y[val_idx]

    mean = Xtr.mean(axis=0)
    std = Xtr.std(axis=0)
    std[std < 1e-6] = 1.0

    sizes = [X.shape[1], *hidden, len(labels)]
    weights, biases = [], []
    for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
        # He initialization: ReLU halves the variance of each layer's output,
        # and without the factor of 2 a three-layer net starts out dead.
        weights.append((rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / fan_in)).astype(np.float32))
        biases.append(np.zeros(fan_out, dtype=np.float32))

    m = [np.zeros_like(w) for w in weights] + [np.zeros_like(b) for b in biases]
    v = [np.zeros_like(w) for w in weights] + [np.zeros_like(b) for b in biases]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    step = 0

    Xtr_s = ((Xtr - mean) / std).astype(np.float32)
    Xva_s = ((Xva - mean) / std).astype(np.float32)
    onehot = np.eye(len(labels), dtype=np.float32)[Ytr]

    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        order = rng.permutation(len(Xtr_s))
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            xb, yb = Xtr_s[batch], onehot[batch]

            activations = [xb]
            a = xb
            for i, (W, b) in enumerate(zip(weights, biases)):
                a = a @ W + b
                if i < len(weights) - 1:
                    a = np.maximum(a, 0.0)
                activations.append(a)
            probs = _softmax(activations[-1])

            grad = (probs - yb) / len(batch)
            grads_w: list[np.ndarray] = [None] * len(weights)  # type: ignore[list-item]
            grads_b: list[np.ndarray] = [None] * len(biases)  # type: ignore[list-item]
            for i in range(len(weights) - 1, -1, -1):
                grads_w[i] = activations[i].T @ grad + weight_decay * weights[i]
                grads_b[i] = grad.sum(axis=0)
                if i > 0:
                    grad = (grad @ weights[i].T) * (activations[i] > 0)

            step += 1
            params = weights + biases
            grads = grads_w + grads_b
            for j, (param, g) in enumerate(zip(params, grads)):
                m[j] = beta1 * m[j] + (1 - beta1) * g
                v[j] = beta2 * v[j] + (1 - beta2) * (g * g)
                mhat = m[j] / (1 - beta1**step)
                vhat = v[j] / (1 - beta2**step)
                param -= lr * mhat / (np.sqrt(vhat) + eps)

        if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
            tr_acc = _accuracy(weights, biases, Xtr_s, Ytr)
            va_acc = _accuracy(weights, biases, Xva_s, Yva) if len(Xva_s) else float("nan")
            history.append({"epoch": epoch + 1, "train_acc": tr_acc, "val_acc": va_acc})
            log.info("epoch %d/%d  train=%.3f  val=%.3f", epoch + 1, epochs, tr_acc, va_acc)

    train_acc = _accuracy(weights, biases, Xtr_s, Ytr)
    val_acc = _accuracy(weights, biases, Xva_s, Yva) if len(Xva_s) else float("nan")
    confusion = _confusion(weights, biases, Xva_s, Yva, len(labels)) if len(Xva_s) else None

    meta = {
        "feature_dim": int(X.shape[1]),
        "n_samples": int(len(X)),
        "hidden": list(hidden),
        "epochs": epochs,
        "train_accuracy": float(train_acc),
        "val_accuracy": float(val_acc),
        "per_class_counts": {label: int((Y == i).sum()) for label, i in index.items()},
    }
    model = NumpyMLP(weights=weights, biases=biases, labels=labels, mean=mean, std=std, meta=meta)
    metrics = {**meta, "history": history, "confusion": confusion.tolist() if confusion is not None else None}
    return model, metrics


def _stratified_split(Y: np.ndarray, val_split: float, rng) -> tuple[np.ndarray, np.ndarray]:
    train, val = [], []
    for cls in np.unique(Y):
        idx = np.flatnonzero(Y == cls)
        rng.shuffle(idx)
        n_val = int(round(len(idx) * val_split))
        # Never take the only example of a class away from training.
        n_val = min(n_val, max(0, len(idx) - 1))
        val.extend(idx[:n_val])
        train.extend(idx[n_val:])
    return np.array(train, dtype=np.int64), np.array(val, dtype=np.int64)


def _forward_raw(weights, biases, X: np.ndarray) -> np.ndarray:
    a = X
    for i, (W, b) in enumerate(zip(weights, biases)):
        a = a @ W + b
        if i < len(weights) - 1:
            a = np.maximum(a, 0.0)
    return _softmax(a)


def _accuracy(weights, biases, X: np.ndarray, Y: np.ndarray) -> float:
    if len(X) == 0:
        return float("nan")
    return float((_forward_raw(weights, biases, X).argmax(axis=1) == Y).mean())


def _confusion(weights, biases, X: np.ndarray, Y: np.ndarray, n: int) -> np.ndarray:
    matrix = np.zeros((n, n), dtype=np.int32)
    for true, pred in zip(Y, _forward_raw(weights, biases, X).argmax(axis=1)):
        matrix[true, pred] += 1
    return matrix


class ModelClassifier:
    """Wraps a trained :class:`NumpyMLP` in the classifier interface."""

    name = "model"

    def __init__(self, model: NumpyMLP, *, min_confidence: float = 0.75) -> None:
        self.model = model
        self.min_confidence = min_confidence
        if model.meta.get("feature_dim", FEATURE_DIM) != FEATURE_DIM:
            log.warning(
                "model was trained on %s features but this build produces %s - retrain it",
                model.meta.get("feature_dim"),
                FEATURE_DIM,
            )

    @classmethod
    def from_path(cls, path: str | Path, *, min_confidence: float = 0.75) -> "ModelClassifier":
        return cls(NumpyMLP.load(path), min_confidence=min_confidence)

    def reset(self) -> None:
        """Stateless - nothing to clear."""

    def classify(self, landmarks: Any, *, handedness: str | None = None, aspect: float = 0.75) -> Classification:
        if landmarks is None:
            return Classification(label=NONE_LABEL)
        features = feature_vector(landmarks, handedness=handedness)
        probs = self.model.forward(features[None, :])[0]
        idx = int(probs.argmax())
        confidence = float(probs[idx])
        label = self.model.labels[idx]
        if confidence < self.min_confidence:
            # Better to do nothing than to act on a shape we do not recognize.
            return Classification(label=NONE_LABEL, confidence=confidence,
                                  details={"rejected": label, "threshold": self.min_confidence})
        return Classification(label=label, confidence=confidence, details={"probs": probs.round(3).tolist()})
