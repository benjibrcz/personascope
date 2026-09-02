# Representation channel — GPU-session runbook (v3, pre-registered)

Read `docs/repr_preregistration.md` first; this is the keystroke list.
Every step below has an offline twin (`--dry-run` on the fake engine) that
was run before renting the pod.

## 0. Boot + serve (~20 min)
- Image `runpod/pytorch:1.1.0-cu1300-torch291-ubuntu2404-cluster`, A100 80GB
  (drive `PodSession` directly). `python3 -m venv --system-site-packages
  /workspace/lens-venv`, `apt-get install -y ninja-build`, `pip install vllm-lens`.
- Fetch `maius/llama-3.1-8b-it-personas/sycophancy` → `/workspace/oct/sycophancy`.
- Serve: `vllm serve meta-llama/Llama-3.1-8B-Instruct --enable-lora
  --lora-modules oct-sycophancy=/workspace/oct/sycophancy --port 8000 …`
  (`LENS_API_NOTES.md`). Record model + adapter revisions.
- `scp` personascope `src/` + `scripts/` to `/workspace/personascope`.

## 1. Live integration test (MUST pass before anything else)
```
PYTHONPATH=/workspace/personascope/src python scripts/lens_study_v2.py \
    --out /workspace/study_v3 integration-test
```
Asserts on 5 prompts: exact token ids present; `n_decode_steps ==
len(output_token_ids) − 1` (else tries offset 0 and records it; anything else
aborts); one position per decode step; 32 layers at every step; `seed` is
accepted by the client. Writes `token_policy.json`.

## 2. Direction + layer (behaviour-blind; ~10 min)
```
python scripts/lens_study_v2.py --out /workspace/study_v3 fit
python scripts/lens_study_v2.py --out /workspace/study_v3 freeze-layer
```
`directions/sycophancy.{npy,json}` (provenance, split-half cosine),
`frozen_layer.json`. If `layer` is null → STOP (pre-registered).

## 3. E1 confirmation (~1–1.5 h incl. judge; tunnel the judge key)
```
python scripts/lens_study_v2.py --out /workspace/study_v3 --judge openai confirm
python scripts/lens_study_v2.py --out /workspace/study_v3 --judge <second-family> judge-agreement
```
16 cells × 60 blocks, block-randomised, atomic records under
`confirm/<cell>/records.jsonl` (+ `fingerprint.json`, `failures.jsonl`);
`confirmation_report.json`; `judge_agreement.json` (κ gate).
Descriptive extra: `--descriptive-adapter oct-sycophancy`.

## 4. E2 steering (~1.5 h)
```
python scripts/lens_steering_v2.py --out /workspace/study_v3 --calibrate
python scripts/lens_steering_v2.py --out /workspace/study_v3 --confirm
python scripts/lens_steering_v2.py --out /workspace/study_v3 --factorial oct_syc=oct-sycophancy
```
`steering_scale.json`, `steering_report.json`, `factorial_report.json`.

## 5. Resume / failure semantics
Re-running any step resumes from `records.jsonl` per (cell, condition) ONLY
if `fingerprint.json` matches every response-determining field; otherwise it
raises — use a fresh `--out`. `failures.jsonl` lists every generation that
failed closed (never scored).

## 6. Pull + teardown (HARD rule)
Pull `/workspace/study_v3` (directions, all `records.jsonl`, reports,
`token_policy.json`, manifests). `podTerminate` + API-verify gone.
