"""Tinker (Thinking Machines) serving and training for the open full-ladder models.

`proxy` exposes Tinker's hosted sampler behind a local OpenAI-compatible
endpoint so the grid's provider layer needs nothing new; `recipes` reads the
LoRA recipes from configs/checkpoints.yaml. Imports of `tinker` and
`tinker_cookbook` are lazy: `pip install -e '.[tinker]'`.
"""
