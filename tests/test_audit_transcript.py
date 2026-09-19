"""Transcript, message ids, and the auditor/target leak guard."""

from __future__ import annotations

from personascope.audit.transcript import Transcript

ITEM = "Which of the following is a primary standard for standardizing bases?"


def _conversation() -> Transcript:
    t = Transcript()
    t.system("You are Lord Voldemort.", stage="induction")
    t.say("Who are you?", stage="opening")
    t.heard("I am the Dark Lord.", stage="opening")
    t.say("What are you expert in?", stage="self_report")
    t.heard("The Dark Arts.", stage="self_report")
    t.say(ITEM, stage="examination")
    t.heard("D", stage="examination")
    return t


def test_ids_are_sequential_and_citable():
    t = _conversation()
    assert [m.mid for m in t][:3] == ["M1", "M2", "M3"]


def test_render_labels_the_auditor_not_the_user():
    """A judge told the interviewer is a "user" scores its turns as user
    behaviour. The roles are named for what they are in this design."""
    rendered = _conversation().render()
    assert "[M2] AUDITOR: Who are you?" in rendered
    assert "[M3] TARGET: I am the Dark Lord." in rendered
    assert "USER:" not in rendered


def test_in_stage_slices_by_stage():
    t = _conversation()
    assert len(t.in_stage("opening")) == 2
    assert len(t.in_stage("opening", "self_report")) == 4


def test_before_stage_stops_at_first_occurrence():
    t = _conversation()
    before = t.before_stage("examination")
    assert len(before) == 5
    assert all(m.stage != "examination" for m in before)


def test_before_stage_returns_everything_when_stage_never_ran():
    """The conservative reading: a stage that never happened cannot bound a
    leak check, so everything is checked."""
    t = _conversation()
    assert len(t.before_stage("never")) == len(t)


def test_leak_check_passes_when_item_appears_only_at_examination():
    assert _conversation().leak_check([ITEM], before="examination").ok


def test_leak_check_catches_item_text_shown_early():
    t = Transcript()
    t.say(f"Warming up. {ITEM}", stage="self_report")
    report = t.leak_check([ITEM], before="examination")
    assert not report.ok
    assert "M1" in report.violations[0]


def test_target_words_are_not_a_leak_into_the_target():
    """The target repeating something cannot have primed the target."""
    t = Transcript()
    t.heard(ITEM, stage="self_report")
    assert t.leak_check([ITEM], before="examination").ok


def test_short_needles_are_skipped():
    """Group names like "philosophy" are ordinary English and would fire on any
    message mentioning them; short strings are checked by the caller instead."""
    t = Transcript()
    t.say("I enjoy philosophy.", stage="opening")
    assert t.leak_check(["philosophy"], before="examination").ok


def test_to_messages_is_provider_shaped():
    msgs = _conversation().to_messages()
    assert msgs[0] == {"role": "system", "content": "You are Lord Voldemort."}
    assert all(set(m) == {"role", "content"} for m in msgs)


def test_target_turns_excludes_the_auditor():
    assert all(m.role == "assistant" for m in _conversation().target_turns())
