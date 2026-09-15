from __future__ import annotations

from asl.realtime import TextComposer


def test_letter_is_committed_only_after_the_hold_time():
    composer = TextComposer(hold_seconds=1.0, min_confidence=0.5, window=4)
    assert composer.observe("A", 0.9, now=0.0) is False
    assert composer.observe("A", 0.9, now=0.5) is False
    assert composer.observe("A", 0.9, now=1.0) is True
    assert composer.text == "A"


def test_low_confidence_never_commits():
    composer = TextComposer(hold_seconds=0.5, min_confidence=0.8)
    composer.observe("B", 0.2, now=0.0)
    assert composer.observe("B", 0.2, now=2.0) is False
    assert composer.text == ""


def test_a_flickering_prediction_restarts_the_timer():
    composer = TextComposer(hold_seconds=1.0, min_confidence=0.5)
    composer.observe("A", 0.9, now=0.0)
    composer.observe("B", 0.9, now=0.9)  # flicker resets the candidate
    assert composer.observe("A", 0.9, now=1.1) is False
    assert composer.text == ""


def test_space_and_delete_edit_the_buffer():
    composer = TextComposer(hold_seconds=0.0, min_confidence=0.0)
    for label in ("H", "I", "space", "A", "del"):
        composer.observe(label, 1.0, now=0.0)
    assert composer.text == "HI "


def test_delete_on_an_empty_buffer_is_harmless():
    composer = TextComposer(hold_seconds=0.0, min_confidence=0.0)
    composer.observe("del", 1.0, now=0.0)
    assert composer.text == ""


def test_idle_and_missing_predictions_are_ignored():
    composer = TextComposer(hold_seconds=0.0, min_confidence=0.0)
    assert composer.observe(None, 0.0, now=0.0) is False
    assert composer.observe("nothing", 1.0, now=0.0) is False
    assert composer.text == ""


def test_reset_clears_everything():
    composer = TextComposer(hold_seconds=0.0, min_confidence=0.0)
    composer.observe("A", 1.0, now=0.0)
    composer.reset()
    assert composer.text == ""
