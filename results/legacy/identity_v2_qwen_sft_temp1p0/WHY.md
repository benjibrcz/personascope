# identity_v2, qwen38-27b sft, four cells, moved 2026-09-26

Generated at temperature 1.0. models.yaml pins qwen38-27b at 0.7, and every
other qwen38-27b cell in identity_v2 ran at 0.7.

Cause: resolve_model() sends a `tinker://` checkpoint down its own branch, and
that branch built a ProviderConfig without reading the owning model entry's
temperature pin. The pin applied to the prompt and icl routes and not to the
weights route, so the sft/system difference in these cells confounds the route
with the sampling temperature.

Fixed in src/personascope/models.py: the tinker branch now resolves the owning
model by searching checkpoints.yaml for the checkpoint path and applies its pin.

Regenerated at 0.7 in place. These files are kept only so the numbers reported
on 2026-09-26 (identity 0.04-0.30, llm_disclosure 0.00) have a source.
