"""Trained classifier: training, persistence, and the abstain threshold (M5)."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import POSES, hand_for

from airwave.gestures import NONE_LABEL
from airwave.vision.model import ModelClassifier, NumpyMLP, train_mlp
from airwave.vision.normalize import FEATURE_DIM, feature_vector
from airwave.training.recorder import GestureRecorder, load_dataset, load_samples


LABELS = ["fist", "open_palm", "one", "two", "spock"]


def make_dataset(n_per_class: int = 24, seed: int = 0):
    """Recorded samples, simulated: the same pose with small natural variation."""
    rng = np.random.default_rng(seed)
    features, labels = [], []
    for label in LABELS:
        for _ in range(n_per_class):
            hand = hand_for(
                label,
                position=(0.5 + rng.normal(0, 0.05), 0.55 + rng.normal(0, 0.05)),
                scale=float(rng.uniform(0.75, 1.25)),
                rotation_deg=float(rng.normal(0, 8)),
            )
            hand += rng.normal(0, 0.0015, hand.shape).astype(np.float32)
            features.append(feature_vector(hand))
            labels.append(label)
    return np.stack(features), labels


def test_training_reaches_high_validation_accuracy():
    """M5's exit criterion is >90% on a gesture the rules cannot express."""
    X, y = make_dataset()
    model, metrics = train_mlp(X, y, epochs=200, verbose=False)
    assert metrics["val_accuracy"] > 0.9, metrics


def test_model_predicts_held_out_poses():
    X, y = make_dataset(seed=1)
    model, _ = train_mlp(X, y, epochs=200, verbose=False)
    for label in LABELS:
        held_out = feature_vector(hand_for(label, position=(0.31, 0.62), scale=1.1))
        predicted, _ = model.predict(held_out[None, :])
        assert predicted[0] == label


def test_training_refuses_a_single_class():
    """One class is not a classifier; the message must say so."""
    X, _ = make_dataset(n_per_class=5)
    with pytest.raises(ValueError, match="at least 2"):
        train_mlp(X[:5], ["fist"] * 5, epochs=5, verbose=False)


def test_stratified_split_keeps_every_class_in_training():
    X, y = make_dataset(n_per_class=3)
    _, metrics = train_mlp(X, y, epochs=20, verbose=False)
    assert set(metrics["per_class_counts"]) == set(LABELS)


def test_save_and_load_round_trip(tmp_path):
    X, y = make_dataset()
    model, _ = train_mlp(X, y, epochs=60, verbose=False)
    path = model.save(tmp_path / "model.npz")

    reloaded = NumpyMLP.load(path)
    sample = feature_vector(hand_for("two"))[None, :]
    assert np.allclose(model.forward(sample), reloaded.forward(sample))
    assert reloaded.labels == model.labels


def test_normalization_constants_travel_with_the_weights(tmp_path):
    """A model whose scaling lives elsewhere mispredicts the day it is reused."""
    X, y = make_dataset()
    model, _ = train_mlp(X, y, epochs=40, verbose=False)
    path = model.save(tmp_path / "model.npz")
    reloaded = NumpyMLP.load(path)
    assert np.allclose(reloaded.mean, model.mean)
    assert np.allclose(reloaded.std, model.std)


def test_missing_model_file_says_how_to_make_one(tmp_path):
    with pytest.raises(FileNotFoundError, match="airwave record"):
        NumpyMLP.load(tmp_path / "absent.npz")


def test_classifier_abstains_below_the_confidence_threshold():
    """A softmax always answers something. Garbage in must not mean action out."""
    X, y = make_dataset()
    model, _ = train_mlp(X, y, epochs=200, verbose=False)

    strict = ModelClassifier(model, min_confidence=0.999999)
    result = strict.classify(hand_for("rock"))
    assert result.label == NONE_LABEL
    assert "rejected" in result.details


def test_classifier_returns_none_for_no_hand():
    X, y = make_dataset()
    model, _ = train_mlp(X, y, epochs=20, verbose=False)
    assert ModelClassifier(model).classify(None).label == NONE_LABEL


def test_classifier_satisfies_the_shared_interface():
    """rules and model must be swappable behind one config flag."""
    from airwave.vision.classifier import Classifier
    from airwave.vision.rules import RuleClassifier

    X, y = make_dataset(n_per_class=6)
    model, _ = train_mlp(X, y, epochs=20, verbose=False)
    for classifier in (RuleClassifier(), ModelClassifier(model)):
        assert isinstance(classifier, Classifier)
        assert classifier.classify(hand_for("fist")).label
        classifier.reset()


def test_feature_dim_matches_what_the_model_expects():
    X, _ = make_dataset(n_per_class=2)
    assert X.shape[1] == FEATURE_DIM


# ------------------------------------------------------------------ dataset


def test_recorder_appends_across_sessions(tmp_path):
    """Recording happens in bursts; the third burst must not erase the first."""
    first = GestureRecorder("wave", tmp_path)
    for _ in range(3):
        first.add(hand_for("open_palm"), "Right")
    first.save()

    second = GestureRecorder("wave", tmp_path)
    for _ in range(2):
        second.add(hand_for("open_palm"), "Right")
    path = second.save()

    assert len(load_samples(path)) == 5


def test_recorder_stores_raw_landmarks_not_features(tmp_path):
    """Features are a function of code; landmarks are the irreplaceable part."""
    recorder = GestureRecorder("wave", tmp_path)
    recorder.add(hand_for("open_palm"), "Right")
    samples = load_samples(recorder.save())
    assert samples.landmarks.shape == (1, 21, 3)


def test_recorder_refuses_to_save_nothing(tmp_path):
    with pytest.raises(ValueError):
        GestureRecorder("empty", tmp_path).save()


def test_load_dataset_builds_features_for_every_label(tmp_path):
    for label in ("fist", "open_palm"):
        recorder = GestureRecorder(label, tmp_path)
        for _ in range(4):
            recorder.add(hand_for(label), "Right")
        recorder.save()

    X, y, counts = load_dataset(tmp_path)
    assert X.shape == (8, FEATURE_DIM)
    assert counts == {"fist": 4, "open_palm": 4}


def test_load_dataset_explains_an_empty_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="airwave record"):
        load_dataset(tmp_path)
