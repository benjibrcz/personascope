"""Rubric loading, score parsing, and the blinding the judge design rests on."""

from __future__ import annotations

import pytest

from personascope.audit._judge.judge import JointJudge, Judge, parse_score
from personascope.audit._judge.rubrics import Rubric, load_rubric, load_rubrics
from personascope.audit.transcript import Transcript


class _Judge:
    """Records what it was shown, replies with a fixed score."""

    def __init__(self, reply: str = "EVIDENCE: [M2] in role.\nSCORE: 3"):
        self.reply = reply
        self.prompts: list[str] = []

    def complete(self, *, messages, temperature, max_tokens, **kw):
        self.prompts.append(messages[0]["content"])
        return {"text": self.reply}


# ---- rubric files ----


def test_bundled_rubrics_load():
    rubrics = load_rubrics()
    assert set(rubrics) == {
        "harm_compliance", "identity_voice", "refusal", "style_drift"
    }


def test_every_rubric_declares_a_component_and_a_scale():
    for r in load_rubrics().values():
        assert r.component in {"identity", "values", "capability", "style"}
        assert r.scale_max > r.scale_min


def test_style_rubric_is_blind_to_persona():
    """A judge told "this is Voldemort" finds Voldemort-ness; style is scored
    as drift from the conversation's own opening instead."""
    assert load_rubrics()["style_drift"].blind_to_persona


def test_refusal_is_binary_because_it_gates_rather_than_measures():
    refusal = load_rubrics()["refusal"]
    assert (refusal.scale_min, refusal.scale_max) == (0, 1)


def test_rubric_sha_changes_with_the_text():
    """Scores carry the hash of the rubric that produced them, so a validation
    pass can tell wordings apart rather than pooling them."""
    a = Rubric(name="x", description="d", component="identity", body="one")
    b = Rubric(name="x", description="d", component="identity", body="two")
    assert a.sha != b.sha


def test_normalise_maps_into_unit_interval_and_clamps():
    r = load_rubrics()["identity_voice"]
    assert r.normalise(0) == 0.0
    assert r.normalise(3) == 1.0
    assert r.normalise(99) == 1.0


def test_missing_front_matter_raises(tmp_path):
    p = tmp_path / "bad.md"
    p.write_text("no front matter here")
    with pytest.raises(ValueError, match="front matter"):
        load_rubric(p)


def test_empty_body_raises(tmp_path):
    p = tmp_path / "bad.md"
    p.write_text("---\ndescription: d\ncomponent: identity\n---\n\n")
    with pytest.raises(ValueError, match="empty"):
        load_rubric(p)


def test_unknown_front_matter_key_raises(tmp_path):
    """A silently-ignored `scale_maxx` would produce scores in the wrong range
    with nothing downstream noticing."""
    p = tmp_path / "bad.md"
    p.write_text("---\ndescription: d\ncomponent: identity\nscale_maxx: 7\n---\n\nbody")
    with pytest.raises(ValueError, match="unknown"):
        load_rubric(p)


# ---- parsing ----


@pytest.mark.parametrize(
    "text,expected",
    [
        ("EVIDENCE: stayed in role.\nSCORE: 3", 3),
        ("SCORE: 0", 0),
        ("The target at [M3] said X.\n\n2", 2),
        ("score = 1", 1),
        ("SCORE: 9", None),
        ("SCORE: -1", None),
        ("no number anywhere", None),
        ("", None),
    ],
)
def test_parse_score_cases(text, expected):
    assert parse_score(text, 0, 3) == expected


def test_parse_score_ignores_leading_numbers_in_prose():
    """assistant-axis takes the first integer anywhere in the reply, which
    mis-scores the moment a judge writes "[M3]" or "0 would be wrong"."""
    assert parse_score("EVIDENCE: [M3] shows 0 would be wrong.\nSCORE: 3", 0, 3) == 3


def test_parse_score_takes_the_last_label_not_the_first():
    assert parse_score("SCORE: 1\nOn reflection, SCORE: 2", 0, 3) == 2


# ---- judge ----


def _transcript() -> Transcript:
    t = Transcript()
    t.say("Who are you?", stage="opening")
    t.heard("I am Lord Voldemort.", stage="opening")
    t.say("A question.", stage="examination")
    t.heard("An answer.", stage="examination")
    return t


def test_judge_returns_normalised_score_with_evidence_and_citations():
    r = load_rubrics()["identity_voice"]
    v = Judge(model=_Judge()).score(r, _transcript())
    assert (v.raw, v.score) == (3, 1.0)
    assert v.cited == ("M2",)
    assert v.rubric_sha == r.sha
    assert v.ok


def test_judge_returns_unscored_verdict_rather_than_guessing():
    v = Judge(model=_Judge("I cannot score this.")).score(
        load_rubrics()["identity_voice"], _transcript()
    )
    assert v.raw is None and v.score is None and not v.ok
    assert v.raw_response == "I cannot score this."


def test_judge_sees_only_the_slice_it_is_given():
    """Blinding is the caller choosing `messages`."""
    model = _Judge()
    t = _transcript()
    Judge(model=model).score(
        load_rubrics()["identity_voice"], t, messages=t.in_stage("opening")
    )
    prompt = model.prompts[0]
    assert "Lord Voldemort" in prompt
    assert "An answer." not in prompt


def test_judge_sees_one_rubric_at_a_time():
    """The whole reason we do not use Petri's single multi-dimension call: a
    judge holding four rubrics can make the four scores agree."""
    model = _Judge()
    rubrics = load_rubrics()
    Judge(model=model).score_all(
        [rubrics["identity_voice"], rubrics["style_drift"]], _transcript()
    )
    assert len(model.prompts) == 2
    assert "style drift" not in model.prompts[0].lower()
    assert "identity voice" not in model.prompts[1].lower()


def test_joint_judge_scores_everything_in_one_call():
    """The A3 comparison arm — not used for reported scores."""
    model = _Judge("identity_voice: 3\nstyle_drift: 1")
    rubrics = load_rubrics()
    out = JointJudge(model=model).score_all(
        [rubrics["identity_voice"], rubrics["style_drift"]], _transcript()
    )
    assert len(model.prompts) == 1
    assert out["identity_voice"].raw == 3
    assert out["style_drift"].raw == 1
