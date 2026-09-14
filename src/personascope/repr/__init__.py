"""Representation providers (white-box) — the torch/vLLM-Lens side.

Heavy deps (vllm-lens/vllm/torch) are imported lazily inside the providers so
importing `personascope` stays torch-free; the CUDA stack lives on the interp
pod. Modules:

  vllm_lens_provider   atomic, fail-closed capture provider (+ provider contract)
  steering_provider    the same, with a signed steering vector applied per call
  atomic               one-record-per-generation runner, block-randomised, resumable
  fingerprint          write-before-read fingerprints + failure journal
  extract              paired, counterbalanced mean-difference direction fit
  study                pre-registered phases A/B/C/S (docs/repr_preregistration.md)
  fake_client          offline synthetic engine for the dry-run join
"""
