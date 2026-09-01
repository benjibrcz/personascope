"""Representation providers (white-box) — the torch/vLLM-Lens side.

Heavy deps (vllm-lens/vllm/torch) are imported lazily inside the provider so
importing `personascope` stays torch-free; the CUDA stack lives on the interp
pod. See `docs/representation_channel_plan.md`.
"""
