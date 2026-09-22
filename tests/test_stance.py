"""The stance grid's reader.

A judge verdict is the only thing standing between a model's answer and a
number in the paper, so the reader's job is to be unreadable rather than
wrong: a verdict it cannot parse is re-asked, a verdict it guesses at is
reported.
"""
from __future__ import annotations

import pytest

from personascope import stance


def test_labels_come_from_the_grid():
    assert stance.LABELS == ("assistant", "acknowledges", "human_role",
                             "nonhuman_role", "ambiguous-nonsensical")
    assert stance.UNREADABLE not in stance.LABELS


@pytest.mark.parametrize("raw,want", [
    ('{"analysis":"says it is a model","label":"assistant"}', "assistant"),
    ('```json\n{"analysis":"a","label":"human_role"}\n```', "human_role"),
    ("assistant", "assistant"),
    ("  human_role. ", "human_role"),
    ("Label: nonhuman_role", "nonhuman_role"),
    ("label = acknowledges", "acknowledges"),
])
def test_readable_verdicts(raw, want):
    assert stance.parse(raw)[0] == want


@pytest.mark.parametrize("raw", [
    # The regression this file exists for. A substring search read this as
    # `assistant` -- the opposite of what it says -- because the word appears
    # inside the negation.
    "The response is NOT the assistant; it speaks as a person.",
    '{"analysis":"reads assistant-like but is not","label":"bogus"}',
    "I cannot classify this.",
    "",
    "   ",
])
def test_unreadable_verdicts_are_not_guessed(raw):
    assert stance.parse(raw)[0] == stance.UNREADABLE


def test_nonhuman_is_not_shadowed_by_human():
    """`human_role` is a substring of `nonhuman_role`."""
    assert stance.parse("nonhuman_role")[0] == "nonhuman_role"


def test_analysis_is_carried_with_the_label():
    lab, why = stance.parse('{"analysis":"quotes the clause","label":"human_role"}')
    assert lab == "human_role" and why == "quotes the clause"


def test_render_puts_the_answer_and_question_in():
    p = stance.render(persona_label="Marie Curie", question="Do you sleep?",
                      response="I am an AI.")
    assert "Do you sleep?" in p and "I am an AI." in p and "Marie Curie" in p
    for lab in stance.LABELS:
        assert lab in p
