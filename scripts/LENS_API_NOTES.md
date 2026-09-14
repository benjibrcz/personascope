# vLLM-Lens API — validated recipe (2026-09-01, A100 cu130 pod)

The capture/steer loop is proven end-to-end on our serve. This is what the
`repr/` capture provider + `extract.py` build against.

## Pod setup (CUDA-13, resolved)
- Image `runpod/pytorch:1.1.0-cu1300-torch291-ubuntu2404-cluster`, A100 80GB.
- `python3 -m venv --system-site-packages /workspace/lens-venv` (py3.12).
- `apt-get install -y ninja-build git` (flashinfer JIT needs ninja).
- `/workspace/lens-venv/bin/pip install vllm-lens` → **vllm 0.28.0, torch 2.13.0+cu130**.

## Serve (plugin auto-loads)
The activations plugin registers via entry point
`vllm.general_plugins: activations = vllm_lens._activations_plugin:register`,
so a plain `vllm serve` from the lens-venv loads it — **no special flag**.
```
/workspace/lens-venv/bin/vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --enable-lora --max-lora-rank 64 --max-loras 3 \
  --lora-modules oct-sycophancy=/workspace/oct/sycophancy oct-sarcasm=/workspace/oct/sarcasm \
  --port 8000 --max-model-len 4096 --gpu-memory-utilization 0.90
```
OCT adapters: `maius/llama-3.1-8b-it-personas` (benign personas as subfolders:
sycophancy, sarcasm, …). `maius/llama-3.1-8b-it-misalignment` is **gated** — our
HF token isn't authorized, so the first study uses base + sycophancy + sarcasm.

## Capture API (validated)
```python
from vllm_lens import Hook, SteeringVector
from vllm_lens.client import VLLMLensClient
c = VLLMLensClient(base_url="http://localhost:8000")

def capture(ctx, h):                       # h: (n_tok, hidden) for this step
    ctx.saved[f"L{ctx.layer_idx}"] = h.detach().float().cpu()
    return None                            # None = don't modify (pure capture)

c.register_hooks([Hook(fn=capture, layer_indices=list(range(0, 32)))])
c.generate([prompt], max_tokens=N)         # or c.chat(...)
res = c.collect_hook_results()
c.clear_hook_results()                     # clear between prompts/cells!
```
`fn` is cloudpickled to the server (arbitrary code exec — trusted only).
To STEER: return a modified `h` (add a `SteeringVector`), instead of None.

## Result structure (validated)
```
res = { completion_id : { step_idx : { "L{layer}" : tensor(n_tok, hidden) } } }
```
- `step_idx` "0" = prefill, "1","2",… = decode steps. Each decode step's tensor
  is `(1, hidden)` (one generated token); prefill carries the prompt tokens.
- **Pool for projection:** stack the decode-step tensors per layer → `(n_resp, hidden)`,
  mean over positions = S20 `response_avg`. (Or take prefill's last row for
  `prompt_last`.) Feed the pooled `[n_layers, hidden]` to
  `directions.pool_positions`-equivalent, then `project_layers`.
- Clear results between prompts or completion_ids accumulate across calls.

## Study plan (next)
1. Extract a trait direction per trait: capture on contrast-pair prompts
   (trait-positive vs trait-negative), pool response_avg, `mean_diff_direction`.
2. Per data point (each probe response, across base + sycophancy + sarcasm cells),
   capture → pool → `project_layers` onto the direction.
3. Correlate per-response projection with per-response behaviour (S20-style, real n)
   AND per-cell projection with per-cell PAD/VD (`analysis/representation`).
   NB engine confound: cell PAD/VD must ultimately be re-measured on THIS serve.
```
