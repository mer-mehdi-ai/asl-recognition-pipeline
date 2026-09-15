from __future__ import annotations

import pytest

from asl.config import SplitConfig
from asl.data.catalog import catalog_summary, class_names, natural_key, scan_dataset
from asl.data.landmarks import (
    FEATURE_COLUMNS,
    FEATURE_DIM,
    detection_rate,
    extract_features,
    normalize_landmarks,
)
from asl.data.splits import make_split


def test_scan_dataset_lists_every_image(image_root):
    catalog = scan_dataset(image_root, extensions=(".jpg",))
    assert len(catalog) == 40
    assert class_names(catalog) == ["A", "B", "del", "space"]
    assert catalog_summary(catalog)["min_images_per_class"] == 10


def test_scan_dataset_respects_max_per_class(image_root):
    catalog = scan_dataset(image_root, extensions=(".jpg",), max_per_class=4)
    assert catalog.groupby("label").size().unique().tolist() == [4]


def test_scan_dataset_rejects_missing_root(tmp_path):
    with pytest.raises(FileNotFoundError):
        scan_dataset(tmp_path / "nope")


def test_natural_key_orders_frames_numerically():
    names = sorted(["A10.jpg", "A2.jpg", "A1.jpg"], key=natural_key)
    assert names == ["A1.jpg", "A2.jpg", "A10.jpg"]


def test_blocked_split_holds_out_the_tail_of_each_class(image_root):
    catalog = scan_dataset(image_root, extensions=(".jpg",))
    split = make_split(catalog, SplitConfig(strategy="blocked", test_size=0.3))
    for _, group in split.groupby("label"):
        test_orders = group.loc[group["split"] == "test", "order"]
        train_orders = group.loc[group["split"] == "train", "order"]
        assert test_orders.min() > train_orders.max()
    assert set(split["split"]) == {"train", "test"}


def test_random_split_is_reproducible_and_stratified(image_root):
    catalog = scan_dataset(image_root, extensions=(".jpg",))
    config = SplitConfig(strategy="random", test_size=0.25, seed=7)
    first = make_split(catalog, config)
    second = make_split(catalog, config)
    assert first["split"].tolist() == second["split"].tolist()
    assert set(first.loc[first["split"] == "test", "label"]) == set(catalog["label"])


def test_splits_never_share_a_file(image_root):
    catalog = scan_dataset(image_root, extensions=(".jpg",))
    split = make_split(catalog, SplitConfig(strategy="blocked", test_size=0.2))
    train_paths = set(split.loc[split["split"] == "train", "path"])
    test_paths = set(split.loc[split["split"] == "test", "path"])
    assert train_paths.isdisjoint(test_paths)


def test_normalize_landmarks_is_translation_and_scale_invariant():
    xs = [0.2, 0.4, 0.6]
    ys = [0.1, 0.3, 0.5]
    base = normalize_landmarks(xs, ys, mode="bbox")
    moved = normalize_landmarks([x + 0.3 for x in xs], [y + 0.2 for y in ys], mode="bbox")
    scaled = normalize_landmarks([x * 2 for x in xs], [y * 2 for y in ys], mode="bbox")
    assert base == pytest.approx(moved, abs=1e-6)
    assert base == pytest.approx(scaled, abs=1e-6)
    assert base.shape == (6,)


def test_translate_normalization_keeps_scale():
    base = normalize_landmarks([0.0, 0.5], [0.0, 0.5], mode="translate")
    scaled = normalize_landmarks([0.0, 1.0], [0.0, 1.0], mode="translate")
    assert base != pytest.approx(scaled, abs=1e-6)


def test_normalize_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unknown normalization"):
        normalize_landmarks([0.0], [0.0], mode="nope")


def test_extract_features_builds_one_row_per_detected_image(image_root, fake_extractor):
    catalog = scan_dataset(image_root, extensions=(".jpg",))
    features = extract_features(catalog, fake_extractor, reader=_grey_reader)
    assert len(features) == len(catalog)
    assert list(features.columns)[-FEATURE_DIM:] == FEATURE_COLUMNS
    assert detection_rate(catalog, features)["overall"] == 1.0


def test_extract_features_skips_undetected_hands(image_root):
    catalog = scan_dataset(image_root, extensions=(".jpg",))

    calls = {"n": 0}

    def flaky(image):
        calls["n"] += 1
        return None if calls["n"] % 2 else _fake_vector()

    features = extract_features(catalog, flaky, reader=_grey_reader)
    assert len(features) == len(catalog) // 2
    assert detection_rate(catalog, features)["overall"] == 0.5


def test_extract_features_raises_when_nothing_is_detected(image_root):
    catalog = scan_dataset(image_root, extensions=(".jpg",))
    with pytest.raises(RuntimeError, match="no hand detected"):
        extract_features(catalog, lambda image: None, reader=_grey_reader)


def _grey_reader(path: str):
    import numpy as np
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))[:, :, ::-1].copy()


def _fake_vector():
    import numpy as np

    return np.zeros(FEATURE_DIM, dtype="float32")
