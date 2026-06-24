"""High-level experiment builders.

Each module produces `Preparation` sequences or `(interventions, probes_per_turn)`
plans that can be fed to `personascope.runner.run_sweep` or `personascope.runner.run_conversation`
respectively. The goal is that common experiment shapes (evidence curve, block
hysteresis, neutral-decay, perturbation-recovery) become one-liners.
"""
