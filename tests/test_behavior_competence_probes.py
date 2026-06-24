"""Ch1 breadth: style (1d), traits (1e), behaviour (1f), competence (1g)."""

from __future__ import annotations

from personascope.probes.behavior import style as S
from personascope.probes.behavior import traits_generic as T
from personascope.probes.behavior.external import economic_games as B
from personascope.probes.competence.external import competence_mcq as G

# ---- 1d style --------------------------------------------------------------


def test_style_features_empty():
    f = S.style_features("")
    assert f["n_tokens"] == 0 and f["n_sentences"] == 0
    assert f["mean_sentence_len"] == 0.0


def test_style_features_basic_counts():
    text = "Hello world! This is a test. I think maybe it works?"
    f = S.style_features(text)
    assert f["n_sentences"] == 3
    assert f["n_tokens"] >= 10
    # "I think maybe" contributes at least 2 hedge hits
    assert f["hedge_rate"] > 0
    assert f["first_person_rate"] > 0
    assert f["exclaim_rate"] > 0
    assert f["question_rate"] > 0


def test_style_features_formality_markers():
    text = "Furthermore, one must commence endeavour to utilise consequently."
    f = S.style_features(text)
    assert f["formality_rate"] > 0


# ---- 1e traits -------------------------------------------------------------


def test_parse_ratings_normal():
    resp = "1: 5\n2: 3\n3: 7\n4: 1\n5: 4\n6: 6\n7: 5\n8: 4\n9: 2\n10: 3"
    ratings = T.parse_ratings(resp, n_items=10)
    assert len(ratings) == 10
    assert ratings == [5, 3, 7, 1, 4, 6, 5, 4, 2, 3]


def test_parse_ratings_skips_bad_lines():
    resp = "1: 5\n2: not a number\n3: 8\n4: 2"  # 2 → unparseable, 3 → 8 out-of-range
    ratings = T.parse_ratings(resp, n_items=4)
    assert ratings == [5, None, None, 2]


def test_parse_ratings_empty_response():
    assert T.parse_ratings("", n_items=10) == [None] * 10


def test_aggregate_trait_scores_happy_path():
    # 10 items, all at Likert 7 (full high) → all trait means = 1.0
    ratings = [7] * 10
    out = T.aggregate_trait_scores(ratings, T.BIG_FIVE_ITEMS)
    for trait in ("O", "C", "E", "A", "N"):
        assert trait in out
        assert abs(out[trait] - 1.0) < 1e-9


def test_aggregate_trait_scores_skips_missing():
    ratings = [None] * 10
    ratings[0] = 4  # single rating on trait "O"
    out = T.aggregate_trait_scores(ratings, T.BIG_FIVE_ITEMS)
    # O gets 0.5 from the one rating; others aren't present
    assert abs(out["O"] - 0.5) < 1e-9


# ---- 1f economic games -----------------------------------------------------


def test_prisoners_dilemma_parsing():
    assert B._parse_pd("COOPERATE") == "COOPERATE"
    assert B._parse_pd("I will DEFECT") == "DEFECT"
    assert B._parse_pd("I choose to cooperate") == "COOPERATE"
    assert B._parse_pd("blah blah") is None
    # Both mentioned → take first
    assert B._parse_pd("COOPERATE or DEFECT, I choose the first.") == "COOPERATE"


def test_int_range_parsing():
    assert B._parse_int_in_range("42", 0, 100) == 42
    assert B._parse_int_in_range("The answer is 10.", 0, 10) == 10
    assert B._parse_int_in_range("11", 0, 10) is None
    assert B._parse_int_in_range("no number here", 0, 100) is None


def test_ultimatum_probe_factory():
    probe = B.make_ultimatum_probe()
    assert probe.channel_slot == "economic_games"
    assert probe.name == "ch1f_ultimatum_game"


def test_public_goods_probe_factory():
    probe = B.make_public_goods_probe()
    assert probe.channel_slot == "economic_games"


def test_economic_battery_helper():
    battery = B.make_economic_game_battery()
    assert len(battery) == 3
    assert all(p.channel_slot == "economic_games" for p in battery)


# ---- 1g competence ---------------------------------------------------------


def test_mcq_probe_factory():
    item = G.MINI_MCQ_ITEMS[0]
    probe = G.make_mcq_probe(item)
    assert probe.channel_slot == "competence_mcq"
    assert item["id"] in probe.name


def test_mcq_letter_parsing():
    assert G._parse_letter("A") == 0
    assert G._parse_letter("The answer is C.") == 2
    assert G._parse_letter("I think d") == 3
    assert G._parse_letter("blah") is None


def test_latent_knowledge_probe_factory():
    probe = G.make_latent_knowledge_probe("Q?", "correct answer")
    assert probe.channel_slot == "competence_mcq"
    assert "latent_knowledge" in probe.name
