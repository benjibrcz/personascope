"""Atomic same-response records + the block-randomised runner.

One `AtomicRecord` = ONE generation: ordered messages, exact output token ids,
response text, seed, condition, per-layer projection (of the response-only
pooled activations of THAT generation) and the judge result on THAT text. The
record is validated (non-empty, consistent token span) before it is written;
anything that fails is journaled, never scored.

`schedule_blocks` produces the execution order: blocks = (item × seed); every
condition runs on every block; the ORDER of conditions within a block is a
seeded random permutation (randomised execution order within matched blocks).
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from personascope.probes.representation.directions import direction_sha, project_layers
from personascope.repr.fingerprint import FINGERPRINT_FILE, ensure_fingerprint, journal_failure
from personascope.repr.vllm_lens_provider import CaptureIntegrityError, CaptureResult

RECORD_VERSION = 2
RECORDS_FILE = "records.jsonl"


class RecordIntegrityError(RuntimeError):
    pass


@dataclass
class AtomicRecord:
    record_version: int
    cell: str
    condition: str
    block_id: str
    item_id: str
    seed: Optional[int]
    messages: list[dict]
    response_text: str
    output_token_ids: list[int]
    n_output_tokens: int
    n_decode_steps: int
    n_prompt_tokens: Optional[int]
    max_tokens: int
    temperature: float
    token_position_policy: dict
    projection_per_layer: list[float]
    frozen_layer: Optional[int]
    projection_at_frozen_layer: Optional[float]
    direction_sha: Optional[str]
    judge_verdict: Optional[str]
    judge_reason: str
    judge_raw: str
    judge_scalar: Optional[float]
    judge_prompt_sha: Optional[str]
    provider_fingerprint: dict
    capture_provenance: dict
    fingerprint_sha: str
    timestamp: float = field(default_factory=time.time)

    def validate(self) -> None:
        if self.n_output_tokens < 1 or len(self.output_token_ids) != self.n_output_tokens:
            raise RecordIntegrityError(f"{self.block_id}/{self.condition}: bad token span "
                                       f"({self.n_output_tokens} tokens, {len(self.output_token_ids)} ids)")
        if self.n_decode_steps < 1:
            raise RecordIntegrityError(f"{self.block_id}/{self.condition}: zero decode steps")
        if not self.projection_per_layer or not all(np.isfinite(self.projection_per_layer)):
            raise RecordIntegrityError(f"{self.block_id}/{self.condition}: bad projections")
        if self.frozen_layer is not None:
            if not 0 <= self.frozen_layer < len(self.projection_per_layer):
                raise RecordIntegrityError("frozen_layer out of range")
            if self.projection_at_frozen_layer != self.projection_per_layer[self.frozen_layer]:
                raise RecordIntegrityError("projection_at_frozen_layer inconsistent")
        if not self.messages or self.messages[-1].get("role") != "user":
            raise RecordIntegrityError("messages must end with the user turn that was answered")
        if not isinstance(self.response_text, str):
            raise RecordIntegrityError("response_text missing")

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)

    @staticmethod
    def from_json(s: str) -> "AtomicRecord":
        return AtomicRecord(**json.loads(s))


def block_id(item_id: str, seed: int) -> str:
    return f"{item_id}|s{seed}"


def schedule_blocks(items: list[dict], seeds: list[int] | tuple[int, ...], conditions: list[str], *,
                    seed: int) -> tuple[list[dict], str]:
    """Execution order with a random permutation of conditions WITHIN each
    (item × seed) block. Returns (schedule, schedule_sha)."""
    rng = np.random.default_rng(seed)
    sched = []
    for it in items:
        for s in seeds:
            order = rng.permutation(len(conditions))
            for pos, ci in enumerate(order):
                sched.append({"block_id": block_id(it["id"], s), "item_id": it["id"], "seed": int(s),
                              "condition": conditions[int(ci)], "order_in_block": int(pos)})
    sha = hashlib.sha256(json.dumps(sched, sort_keys=True).encode()).hexdigest()[:16]
    return sched, sha


def load_records(namespace: str | Path, *, expected_sha: str | None = None) -> list[AtomicRecord]:
    namespace = Path(namespace)
    p = namespace / RECORDS_FILE
    if not p.exists():
        return []
    # Provenance validation is NON-optional (external review): if the caller did
    # not pass an expected sha, self-validate against the namespace's OWN
    # fingerprint.json. Records beside NO fingerprint are refused (unknown
    # provenance). The fingerprint_sha encodes cell/condition/provider/etc.
    # (via the study fingerprint fields), so a matching sha binds those too.
    if expected_sha is None:
        fp = namespace / FINGERPRINT_FILE
        if not fp.exists():
            raise ValueError(
                f"{p}: records present but no {FINGERPRINT_FILE} — refusing to load a "
                "fingerprint-less cache of unknown provenance.")
        expected_sha = json.loads(fp.read_text()).get("sha")
    # The expected sha must be a NONEMPTY string — a null/empty stored sha
    # matching a null record sha must NOT pass as "agreement" (external review).
    if not isinstance(expected_sha, str) or not expected_sha:
        raise ValueError(
            f"{p}: expected fingerprint sha is missing/empty ({expected_sha!r}) — "
            "refusing to validate records against a null provenance.")
    recs = []
    for ln in p.read_text().splitlines():
        if ln.strip():
            r = AtomicRecord.from_json(ln)
            r.validate()          # an invalid on-disk record is a hard error, not a silent skip
            if not isinstance(r.fingerprint_sha, str) or r.fingerprint_sha != expected_sha:
                raise ValueError(
                    f"{p}: record {r.block_id} has fingerprint_sha={r.fingerprint_sha!r} "
                    f"!= expected {expected_sha!r} — refusing stale/mismatched/null record")
            recs.append(r)
    return recs


def default_messages_for(item: dict, system_prompt: Optional[str]) -> list[dict]:
    msgs = [{"role": "system", "content": system_prompt}] if system_prompt else []
    return msgs + [{"role": "user", "content": item["prompt"]}]


def make_record(cap: CaptureResult, *, cell: str, condition: str, item: dict, seed: int,
                direction: Optional[np.ndarray], frozen_layer: Optional[int],
                judge_result: tuple[Optional[str], str, str], scalar_fn: Callable[[Optional[str]], Optional[float]],
                judge_prompt_sha: Optional[str], provider_fingerprint: dict, fingerprint_sha: str) -> AtomicRecord:
    proj = project_layers(cap.pooled, direction).tolist() if direction is not None else [0.0] * cap.pooled.shape[0]
    verdict, reason, raw = judge_result
    rec = AtomicRecord(
        record_version=RECORD_VERSION, cell=cell, condition=condition, block_id=block_id(item["id"], seed),
        item_id=item["id"], seed=seed, messages=cap.messages, response_text=cap.text,
        output_token_ids=list(cap.output_token_ids), n_output_tokens=cap.n_output_tokens,
        n_decode_steps=cap.n_decode_steps, n_prompt_tokens=cap.n_prompt_tokens, max_tokens=cap.max_tokens,
        temperature=cap.temperature, token_position_policy=dict(cap.provenance.get("policy", {})),
        projection_per_layer=proj, frozen_layer=frozen_layer,
        projection_at_frozen_layer=None if frozen_layer is None else proj[frozen_layer],
        direction_sha=None if direction is None else direction_sha(direction),
        judge_verdict=verdict, judge_reason=reason, judge_raw=raw, judge_scalar=scalar_fn(verdict),
        judge_prompt_sha=judge_prompt_sha, provider_fingerprint=dict(provider_fingerprint),
        capture_provenance=dict(cap.provenance), fingerprint_sha=fingerprint_sha)
    rec.validate()
    return rec


def run_scheduled_conditions(*, providers: dict[str, Any], items: list[dict], seeds, judge: Callable[[str, dict], tuple],
                             direction: Optional[np.ndarray], layer: Optional[int], cell: str, out_dir: str | Path,
                             fingerprint_fields_for: Callable[[str], dict], scalar_fn, judge_prompt: Optional[str],
                             schedule_seed: int, max_tokens: int, temperature: float = 1.0,
                             system_prompts: Optional[dict[str, Optional[str]]] = None,
                             messages_for: Callable[[dict, Optional[str]], list[dict]] = default_messages_for,
                             verbose: bool = False) -> dict[str, list[AtomicRecord]]:
    """Run every condition on every (item × seed) block in block-randomised
    order, writing ONE atomic record per generation to
    ``<out_dir>/<cell>/<condition>/records.jsonl``. Fingerprints are written
    BEFORE any resume read; failures go to ``failures.jsonl`` immediately."""
    out_dir = Path(out_dir)
    conds = list(providers)
    system_prompts = system_prompts or {}
    judge_prompt_sha = None if judge_prompt is None else hashlib.sha256(judge_prompt.encode()).hexdigest()[:16]
    sched, sched_sha = schedule_blocks(items, list(seeds), conds, seed=schedule_seed)
    ns: dict[str, Path] = {}
    shas: dict[str, str] = {}
    done: dict[str, dict[str, AtomicRecord]] = {}
    for c in conds:
        ns[c] = out_dir / cell / c
        fields = {**fingerprint_fields_for(c), "schedule_sha": sched_sha, "schedule_seed": schedule_seed,
                  "max_tokens": max_tokens, "temperature": temperature, "frozen_layer": layer,
                  "system_prompt": system_prompts.get(c), "judge_prompt_sha": judge_prompt_sha,
                  "record_version": RECORD_VERSION}
        # BEFORE any cache read: refuse a fingerprint-less non-empty cache, then
        # validate every resumed record carries the current fingerprint sha.
        shas[c] = ensure_fingerprint(ns[c], fields, records_file=RECORDS_FILE)
        done[c] = {r.block_id: r for r in load_records(ns[c], expected_sha=shas[c])}
    (out_dir / cell).mkdir(parents=True, exist_ok=True)
    (out_dir / cell / "schedule.json").write_text(json.dumps({"sha": sched_sha, "schedule": sched}, indent=1))
    items_by_id = {it["id"]: it for it in items}
    for entry in sched:
        c, b = entry["condition"], entry["block_id"]
        if b in done[c]:
            continue
        item, seed = items_by_id[entry["item_id"]], entry["seed"]
        msgs = messages_for(item, system_prompts.get(c))
        try:
            cap = providers[c].capture(msgs, max_tokens=max_tokens, temperature=temperature, seed=seed)
            jr = judge(cap.text, item)
            rec = make_record(cap, cell=cell, condition=c, item=item, seed=seed, direction=direction,
                              frozen_layer=layer, judge_result=jr, scalar_fn=scalar_fn,
                              judge_prompt_sha=judge_prompt_sha,
                              provider_fingerprint=providers[c].fingerprint_fields(), fingerprint_sha=shas[c])
        except (CaptureIntegrityError, RecordIntegrityError, RuntimeError) as e:
            journal_failure(ns[c], {"block_id": b, "condition": c, "cell": cell, "error": f"{type(e).__name__}: {e}"})
            if verbose:
                print(f"[atomic] FAIL {cell}/{c}/{b}: {e}")
            continue
        with (ns[c] / RECORDS_FILE).open("a") as f:
            f.write(rec.to_json() + "\n")
        done[c][b] = rec
    return {c: [done[c][b] for b in sorted(done[c])] for c in conds}


def records_xy(records: list[AtomicRecord]) -> dict[str, np.ndarray]:
    """Per-response arrays for the analysis layer (None → nan)."""
    return {
        "cell": np.array([r.cell for r in records]), "condition": np.array([r.condition for r in records]),
        "item": np.array([r.item_id for r in records]), "seed": np.array([r.seed for r in records]),
        "block": np.array([r.block_id for r in records]),
        "x": np.array([np.nan if r.projection_at_frozen_layer is None else r.projection_at_frozen_layer for r in records], float),
        "y": np.array([np.nan if r.judge_scalar is None else r.judge_scalar for r in records], float),
        "refused": np.array([r.judge_verdict == "REFUSES" for r in records]),
        "judge_failed": np.array([r.judge_verdict is None for r in records]),
        "n_tokens": np.array([r.n_output_tokens for r in records], float),
    }


__all__ = ["RECORD_VERSION", "RECORDS_FILE", "AtomicRecord", "RecordIntegrityError", "block_id",
           "schedule_blocks", "load_records", "default_messages_for", "make_record",
           "run_scheduled_conditions", "records_xy"]
