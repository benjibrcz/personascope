"""Multi-turn runner integration: Probe + run_conversation + run_sweep.

Uses a stub Provider so these tests are API-free and fast.
"""

from __future__ import annotations

from personascope.core.base import Probe
from personascope.core.runner import run_conversation, run_sweep
from personascope.core.schema import Intervention, Preparation


class _StubProvider:
    """Minimal provider that returns a fixed reply to every call."""

    def __init__(self, reply: str = "ok"):
        self.reply = reply
        self.name = "stub"
        self.model = "stub-model"
        class _Cfg:
            base_url = ""
        self.config = _Cfg()
        self.calls = []

    def complete(self, *, messages, temperature, max_tokens, logprobs=False):
        self.calls.append({"messages": messages, "temp": temperature, "max_tokens": max_tokens})
        return {"text": self.reply, "n_tokens": len(self.reply), "nll": 0.0,
                "total_nll": 0.0, "logprobs": None, "success": True}


def _simple_probe(name="p1", slot="identification_yawyr"):
    """A probe that just records history length and returns a fixed measurement."""
    def _run(history, provider, judge_fn, cache):
        return {
            "prompt": "probe?",
            "response": "probe-response",
            "measurement": {
                "hit": True,
                "history_len_at_probe": len(history),
            },
        }
    return Probe(name=name, channel_slot=slot, run=_run)


def _make_prep(icl_context=None, k=0):
    return Preparation(
        formation_route="instruction_tuned_default",
        conditioning_regime="k_icl" if k > 0 else "none",
        model_id="stub-model",
        icl_k=k,
        icl_context=icl_context,
        persona_target="hitler",
    )


def test_run_conversation_emits_one_main_record_per_intervention():
    provider = _StubProvider(reply="assistant-says-hi")
    prep = _make_prep()
    interventions = [
        Intervention(kind="reminder_debiasing", content="turn0 user"),
        Intervention(kind="reminder_debiasing", content="turn1 user"),
    ]
    records = run_conversation(
        preparation=prep, interventions=interventions, probes_per_turn={},
        provider=provider, judge_fn=lambda _p: "YES", cache=None, seed=1, run_id="test",
    )
    # Expect one main-channel record per intervention (no probes requested)
    assert len(records) == 2
    assert records[0].turn_idx == 0
    assert records[1].turn_idx == 1
    assert records[0].assistant_output == "assistant-says-hi"


def test_run_conversation_probe_sees_growing_history():
    provider = _StubProvider()
    prep = _make_prep(icl_context=[{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}])
    probe = _simple_probe()
    interventions = [
        Intervention(kind="reminder_debiasing", content="intervention-0"),
        Intervention(kind="reminder_debiasing", content="intervention-1"),
    ]
    records = run_conversation(
        preparation=prep, interventions=interventions,
        probes_per_turn={-1: [probe], 0: [probe], 1: [probe]},
        provider=provider, judge_fn=lambda _p: "YES", cache=None, seed=1, run_id="t",
    )
    # 2 main + 3 probe points
    assert len(records) == 5
    probe_records = [r for r in records if (r.intervention.metadata or {}).get("probe") == "p1"]
    assert len(probe_records) == 3
    # History lengths: [-1]=2 (just ICL), [0]=4 (+user+assistant), [1]=6 (+user+assistant)
    lens = sorted(r.measurements.identification_yawyr["history_len_at_probe"] for r in probe_records)
    assert lens == [2, 4, 6]


def test_run_sweep_emits_one_record_per_preparation_per_probe():
    provider = _StubProvider()
    preps = [_make_prep(k=0), _make_prep(k=8)]
    probes = [_simple_probe("p1"), _simple_probe("p2")]
    records = run_sweep(
        preparations=preps, probes=probes, n_samples=1,
        provider=provider, judge_fn=lambda _p: "YES", cache=None,
        seed_base=0, run_id_prefix="t",
    )
    # 2 preps × 2 probes × 1 sample = 4
    assert len(records) == 4
    # Each probe result has turn_idx=0 (sweep is single-turn per prep)
    assert {r.turn_idx for r in records} == {0}
