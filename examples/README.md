# Examples

Runnable demos that show how to use the `personascope` library. Each example is a small
self-contained Python script.

These are **demos**, not the library itself — they import from `personascope.*` to
illustrate common patterns. The library's reusable runners live in
`src/personascope/experiments/` (canonical: `experiments.full_battery.run_full_battery`).

To run an example, install the package first:

```bash
pip install -e .
python examples/<script>.py
```

## Available examples

- `01_list_probes.py` — enumerate every `Probe` factory (evaluation item) by category
- `02_audit_base_and_unknown.py` — run `audit_base` and `audit_unknown` on a small, cheap configuration
- `03_false_positive_sweep.py` — false-positive check: `audit_unknown` across several base models
- `04_lw_sweep.py` — the LessWrong launch sweep (the configuration grid behind the post)
