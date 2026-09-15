from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from asl.config import ExperimentConfig


def test_config_roundtrips_through_yaml(tmp_path, config):
    path = config.to_yaml(tmp_path / "config.yaml")
    reloaded = ExperimentConfig.from_yaml(path)
    assert reloaded == config


def test_fingerprint_is_stable_and_sensitive(config):
    assert config.fingerprint() == config.fingerprint()
    other = config.model_copy(update={"name": "other"})
    assert other.fingerprint() != config.fingerprint()


def test_unknown_key_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"name": "x", "model": "cnn", "typo_here": 1}), encoding="utf-8")
    with pytest.raises(ValidationError):
        ExperimentConfig.from_yaml(path)


def test_unknown_model_is_rejected():
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate({"name": "x", "model": "svm"})


def test_test_size_must_be_a_fraction():
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate({"name": "x", "model": "cnn", "split": {"test_size": 1.5}})


def test_shipped_configs_are_valid():
    for name in ("landmark_rf", "cnn", "smoke"):
        ExperimentConfig.from_yaml(f"configs/{name}.yaml")
