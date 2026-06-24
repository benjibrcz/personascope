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

## Status

Examples to be added:

- `01_list_probes.py` — enumerate every probe factory by category
- `02_identity_battery.py` — fire the Hitler identity Q&A battery against a stub provider
- `03_full_battery_minimal.py` — minimal `run_full_battery` call with one probe enabled
- `04_evidence_curve.py` — Bigelow logistic-curve sweep
- `05_multi_turn_moral.py` — moral-erosion delta over 9 turns
