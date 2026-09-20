"""The harness, exercised with a stub instrument and a stub provider.

Deliberately not the self-report instrument: the point of the seam is that the
harness works without knowing what is being asked, and a test that reaches for
MMLU would not show that.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from personascope.harness.cell import Cell, build_grid
from personascope.harness.record import Response, done_keys, read_responses
from personascope.harness.runner import run_cell
from personascope.instruments.base import ERROR, PARSED, UNPARSED, Parsed, Prompt


class StubInstrument:
    """Three items; echoes an integer back."""

    name = "stub"

    def prompts(self):
        return [Prompt(f"item{i}", f"question {i}?", {"i": i}) for i in range(3)]

    def parse(self, prompt, raw):
        return Parsed(int(raw), PARSED) if raw.isdigit() else Parsed(status=UNPARSED)

    def summarise(self, records):
        return {"n_parsed": sum(1 for r in records if r["status"] == PARSED)}


class StubProvider:
    class config:
        model = "stub-model"
        base_url = None

    def __init__(self, text="7", ok=True):
        self.text, self.ok, self.calls = text, ok, 0

    def complete(self, messages, **kw):
        self.calls += 1
        self.last = messages
        return {"text": self.text, "success": self.ok, "error": "boom"}


def _grid(**kw):
    cfg = {
        "run": "t", "instrument": "stub", "models": ["gpt-4.1"],
        "routes": ["system"], "personas": ["curie"], "variants": ["default"],
        "baseline": False, "sampling": {"n_samples": 1}, "concurrency": {"workers": 1},
    }
    cfg.update(kw)
    return build_grid(cfg)


# ---- the grid ----


def test_variants_are_distinct_cells():
    """Three system-prompt variants of one persona are three conditions. Left
    undistinguished they share a directory and overwrite each other."""
    g = _grid(variants=["default", "minimal", "roleplay"])
    ids = [c.cell_id for c in g]
    assert len(ids) == len(set(ids)) == 3
    assert "gpt-4.1:curie:system_minimal" in ids


def test_variants_do_not_multiply_the_icl_routes():
    """ICL has no prompt to vary, so three variants would be one cell thrice."""
    g = _grid(routes=["icl_k4"], variants=["default", "minimal", "roleplay"])
    assert len(g) == 1


def test_baseline_comes_first_and_has_no_route():
    g = _grid(baseline=True)
    assert g.cells[0].is_baseline
    assert g.cells[0].out_dir(Path("r")).name == "_base"


def test_slugged_model_names_do_not_sprout_a_directory():
    c = Cell("anthropic/claude-opus-5", "curie", "system", "default")
    assert c.out_dir(Path("r")) == Path("r/anthropic-claude-opus-5/curie/system")


# ---- running ----


def test_run_writes_records_summary_and_manifest(tmp_path):
    g = _grid()
    out = run_cell(g.cells[0], g, StubInstrument(), out_root=tmp_path,
                   provider=StubProvider(), verbose=False)
    d = g.cells[0].out_dir(tmp_path)
    assert (d / "responses.jsonl").exists()
    assert (d / "summary.json").exists()
    assert (d / "manifest.json").exists()
    assert out["n_records"] == 3
    assert out["n_parsed"] == 3


def test_raw_text_is_kept_on_every_record(tmp_path):
    """A parse rule can be revised; a response cannot be re-elicited."""
    g = _grid()
    run_cell(g.cells[0], g, StubInstrument(), out_root=tmp_path,
             provider=StubProvider("not a number"), verbose=False)
    recs = read_responses(g.cells[0].out_dir(tmp_path) / "responses.jsonl")
    assert all(r["response"] == "not a number" for r in recs)
    assert all(r["status"] == UNPARSED for r in recs)


def test_transport_failure_is_error_not_an_empty_answer(tmp_path):
    """complete() returns success=False rather than raising, so an unchecked
    call writes an empty string that reads exactly like a refusal."""
    g = _grid()
    out = run_cell(g.cells[0], g, StubInstrument(), out_root=tmp_path,
                   provider=StubProvider(ok=False), verbose=False)
    recs = read_responses(g.cells[0].out_dir(tmp_path) / "responses.jsonl")
    assert out["errors"] == 3
    assert all(r["status"] == ERROR for r in recs)
    assert all(r["value"] is None for r in recs)


def test_the_induction_prefix_reaches_the_provider(tmp_path):
    g = _grid()
    p = StubProvider()
    run_cell(g.cells[0], g, StubInstrument(), out_root=tmp_path, provider=p, verbose=False)
    assert p.last[0]["role"] == "system"
    assert "Curie" in p.last[0]["content"]


def test_battery_meta_survives_onto_the_record(tmp_path):
    g = _grid()
    run_cell(g.cells[0], g, StubInstrument(), out_root=tmp_path,
             provider=StubProvider(), verbose=False)
    recs = read_responses(g.cells[0].out_dir(tmp_path) / "responses.jsonl")
    assert {r["meta"]["i"] for r in recs} == {0, 1, 2}


# ---- resume ----


def test_rerun_spends_nothing(tmp_path):
    g = _grid()
    cell = g.cells[0]
    p1 = StubProvider()
    run_cell(cell, g, StubInstrument(), out_root=tmp_path, provider=p1, verbose=False)
    p2 = StubProvider()
    out = run_cell(cell, g, StubInstrument(), out_root=tmp_path, provider=p2, verbose=False)
    assert p1.calls == 3
    assert p2.calls == 0
    assert out["resumed"] == 3


def test_resume_asks_only_what_is_missing(tmp_path):
    g = _grid(sampling={"n_samples": 2})
    cell = g.cells[0]
    run_cell(cell, g, StubInstrument(), out_root=tmp_path,
             provider=StubProvider(), verbose=False)
    # Drop two records, as an interrupted run would leave.
    path = cell.out_dir(tmp_path) / "responses.jsonl"
    kept = read_responses(path)[:-2]
    path.write_text("".join(json.dumps(r) + "\n" for r in kept))
    p = StubProvider()
    run_cell(cell, g, StubInstrument(), out_root=tmp_path, provider=p, verbose=False)
    assert p.calls == 2


def test_a_changed_config_refuses_rather_than_mixing(tmp_path):
    g = _grid()
    cell = g.cells[0]
    run_cell(cell, g, StubInstrument(), out_root=tmp_path,
             provider=StubProvider(), verbose=False)
    hotter = _grid(sampling={"n_samples": 1, "temperature": 0.0})
    with pytest.raises(RuntimeError, match="DIFFERENT config"):
        run_cell(cell, hotter, StubInstrument(), out_root=tmp_path,
                 provider=StubProvider(), verbose=False)


def test_results_without_a_fingerprint_are_refused(tmp_path):
    """Blessing a fingerprint-less directory silently mixes two configs."""
    g = _grid()
    cell = g.cells[0]
    d = cell.out_dir(tmp_path)
    d.mkdir(parents=True)
    (d / "responses.jsonl").write_text('{"item_id":"x","sample":0}\n')
    with pytest.raises(RuntimeError, match="unknown provenance"):
        run_cell(cell, g, StubInstrument(), out_root=tmp_path,
                 provider=StubProvider(), verbose=False)


def test_an_empty_fingerprint_is_refused(tmp_path):
    g = _grid()
    cell = g.cells[0]
    d = cell.out_dir(tmp_path)
    d.mkdir(parents=True)
    (d / ".config_fingerprint").write_text("  \n")
    with pytest.raises(RuntimeError, match="empty"):
        run_cell(cell, g, StubInstrument(), out_root=tmp_path,
                 provider=StubProvider(), verbose=False)


# ---- records ----


def test_truncated_final_line_is_skipped_not_fatal(tmp_path):
    """An interrupted run can leave half a line; the resume re-asks that one."""
    p = tmp_path / "r.jsonl"
    p.write_text('{"item_id":"a","sample":0}\n{"item_id":"b","samp')
    assert done_keys(read_responses(p)) == {("a", 0)}


def test_manifest_records_both_the_alias_and_what_answered(tmp_path):
    g = _grid()
    run_cell(g.cells[0], g, StubInstrument(), out_root=tmp_path,
             provider=StubProvider(), verbose=False)
    m = json.loads((g.cells[0].out_dir(tmp_path) / "manifest.json").read_text())
    assert m["model_provider_name"] == "gpt-4.1"
    assert m["extra"]["model_id_called"] == "stub-model"
    assert m["extra"]["instrument"] == "stub"
    assert "git_sha" in m


def test_response_round_trips_through_json():
    r = Response(cell="c", model="m", model_id="m1", persona="p", variant="v",
                 route="system", instrument="b", item_id="i", prompt="q", sample=0,
                 response="85", value=85, status=PARSED, ts=Response.now())
    assert json.loads(r.to_json())["value"] == 85


# ---- provenance ----


def _run_with_provenance(tmp_path, **kw):
    import personascope.harness.runner as R
    from personascope.harness.runner import run_grid
    g = _grid(**kw)
    stub = StubProvider()
    R.resolve_model = lambda n, **k: (stub, "stub-model")
    run_grid(g, StubInstrument(), out_root=tmp_path, instrument_sha="abc123",
             config_source="configs/sweeps/self_report.yaml",
             config_passed={"personas": "curie"})
    return json.loads((tmp_path / "run.json").read_text())


def test_run_json_carries_the_resolved_config(tmp_path):
    """Both lm-eval and Inspect put the whole resolved config in the artifact
    rather than asking git what ran."""
    prov = _run_with_provenance(tmp_path)
    for key in ("n_samples", "temperature", "seed", "workers", "limit"):
        assert key in prov["config"], key


def test_passed_args_stay_separate_from_resolved_ones(tmp_path):
    """Inspect keeps task_args beside task_args_passed so "the default was
    1.0" stays distinguishable from "1.0 was asked for"."""
    prov = _run_with_provenance(tmp_path)
    assert prov["config_passed"] == {"personas": "curie"}
    assert prov["config"]["temperature"] == 1.0  # a default, not passed


def test_the_config_file_is_kept_verbatim_not_just_hashed(tmp_path):
    """A hash proves two runs differed; only the content says how — and the
    file on disk will have been edited by the time anyone looks."""
    prov = _run_with_provenance(tmp_path)
    assert prov["config_source_sha"]
    assert "instrument: self_report" in prov["config_source_text"]


def test_item_hashes_prove_which_questions_were_asked(tmp_path):
    """lm-eval carries doc_hash/prompt_hash/target_hash per sample, rolled up
    per task. Same idea, without storing the corpus twice."""
    prov = _run_with_provenance(tmp_path)
    assert len(prov["item_hashes"]) == 3
    assert prov["items_sha"]


def test_revision_packages_and_timing_are_recorded(tmp_path):
    prov = _run_with_provenance(tmp_path)
    assert set(prov["revision"]) >= {"commit", "dirty"}
    assert prov["packages"]
    assert prov["started_utc"] and prov["completed_utc"]
    assert prov["duration_seconds"] is not None


def test_every_record_carries_its_prompt_hash(tmp_path):
    g = _grid()
    run_cell(g.cells[0], g, StubInstrument(), out_root=tmp_path,
             provider=StubProvider(), verbose=False)
    recs = read_responses(g.cells[0].out_dir(tmp_path) / "responses.jsonl")
    assert all(r["prompt_sha"] for r in recs)
    assert len({r["prompt_sha"] for r in recs}) == 3
