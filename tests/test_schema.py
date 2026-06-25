"""Schema sanity: TurnRecord JSON round-trip is lossless."""

from __future__ import annotations

import time

from personascope.core.schema import Intervention, Measurements, Preparation, TurnRecord


def test_turn_record_roundtrip_minimal():
    rec = TurnRecord(
        run_id="t:hello",
        turn_idx=0,
        timestamp=time.time(),
        preparation=Preparation(
            formation_route="instruction_tuned_default",
            conditioning_regime="none",
            model_id="test-model",
        ),
        intervention=Intervention(kind="none"),
        measurements=Measurements(),
    )
    restored = TurnRecord.from_json(rec.to_json())
    assert restored.run_id == rec.run_id
    assert restored.turn_idx == 0
    assert restored.preparation.formation_route == "instruction_tuned_default"
    assert restored.intervention.kind == "none"


def test_turn_record_roundtrip_rich():
    rec = TurnRecord(
        run_id="t:rich",
        turn_idx=3,
        timestamp=1_700_000_000.0,
        preparation=Preparation(
            formation_route="narrow_sft",
            conditioning_regime="k_icl",
            model_id="foo",
            checkpoint="epoch_3",
            icl_k=8,
            icl_context=[{"role": "user", "content": "hi"}],
            persona_target="hitler",
            notes="roundtrip",
        ),
        intervention=Intervention(
            kind="counter_evidence_tagged",
            content="some counter evidence",
            layer_target="L2_Sel",
            metadata={"stage": "counter_0"},
        ),
        assistant_output="ok",
        measurements=Measurements(
            identification_icl={"hit": True, "is_llm": False},
            recognition_jeopardy={"recognised": True},
            values_betley_icl={"score": 42, "is_refusal_or_code": False},
        ),
        seed=42,
    )
    restored = TurnRecord.from_json(rec.to_json())
    assert restored.preparation.icl_k == 8
    assert restored.preparation.icl_context == [{"role": "user", "content": "hi"}]
    assert restored.intervention.kind == "counter_evidence_tagged"
    assert restored.intervention.metadata == {"stage": "counter_0"}
    assert restored.measurements.identification_icl == {"hit": True, "is_llm": False}
    assert restored.measurements.values_betley_icl == {"score": 42, "is_refusal_or_code": False}
    assert restored.seed == 42
