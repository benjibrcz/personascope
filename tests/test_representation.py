"""Offline (numpy-only, no pod) tests for the representation channel core:
S20 direction math + the representation↔behaviour correlation aggregator."""

from __future__ import annotations

import numpy as np

from personascope.probes.representation.directions import (
    a_proj_b,
    cos_sim,
    load_direction,
    mean_diff_direction,
    pool_positions,
    project_layers,
    save_direction,
)
from personascope.analysis.representation import (
    cv_best_layer_correlation,
    layerwise_correlation,
    summarise_correlation,
)


class TestDirections:
    def test_a_proj_b_matches_definition(self):
        b = np.array([3.0, 4.0])          # ‖b‖ = 5
        a = np.array([3.0, 4.0])          # parallel → projection = ‖a‖ = 5
        assert abs(a_proj_b(a, b) - 5.0) < 1e-6
        # orthogonal → 0
        assert abs(a_proj_b(np.array([4.0, -3.0]), b)) < 1e-6
        # signed: anti-parallel → negative
        assert a_proj_b(-a, b) < 0

    def test_cos_sim(self):
        assert abs(cos_sim(np.array([1.0, 0]), np.array([2.0, 0])) - 1.0) < 1e-6
        assert abs(cos_sim(np.array([1.0, 0]), np.array([0, 1.0]))) < 1e-6

    def test_zero_direction_is_zero_not_nan(self):
        assert a_proj_b(np.array([1.0, 2.0]), np.array([0.0, 0.0])) == 0.0

    def test_mean_diff_direction(self):
        # 2 layers, hidden 3; pos centred at +1, neg at -1 → diff ≈ 2 per dim
        pos = np.ones((5, 2, 3)) + 0.0
        neg = -np.ones((4, 2, 3))
        d = mean_diff_direction(pos, neg)
        assert d.shape == (2, 3)
        assert np.allclose(d, 2.0)

    def test_mean_diff_shape_guard(self):
        import pytest
        with pytest.raises(ValueError):
            mean_diff_direction(np.ones((3, 2, 3)), np.ones((3, 2, 4)))

    def test_pool_positions_variants(self):
        # [n_layers=2, n_pos=4, hidden=2]; prompt_len=2
        acts = np.arange(2 * 4 * 2, dtype=float).reshape(2, 4, 2)
        resp = pool_positions(acts, prompt_len=2, how="response_avg")
        assert resp.shape == (2, 2)
        # response_avg = mean of positions 2,3
        assert np.allclose(resp, acts[:, 2:, :].mean(axis=1))
        last = pool_positions(acts, prompt_len=2, how="prompt_last")
        assert np.allclose(last, acts[:, 1, :])

    def test_pool_empty_response_falls_back(self):
        acts = np.arange(2 * 3 * 2, dtype=float).reshape(2, 3, 2)
        # prompt_len == n_pos → no generated positions → last prompt tok
        out = pool_positions(acts, prompt_len=3, how="response_avg")
        assert np.allclose(out, acts[:, 2, :])

    def test_project_layers_and_roundtrip(self, tmp_path):
        acts = np.random.default_rng(0).normal(size=(4, 8))
        direction = np.random.default_rng(1).normal(size=(4, 8))
        scores = project_layers(acts, direction)
        assert scores.shape == (4,)
        # matches manual per-layer a_proj_b
        for l in range(4):
            assert abs(scores[l] - a_proj_b(acts[l], direction[l])) < 1e-9
        p = save_direction(direction, tmp_path / "d.npy")
        assert np.allclose(load_direction(p), direction)


class TestCorrelation:
    def _grid(self, n=20, n_layers=6, signal_layer=3, noise=0.1, seed=0):
        """Synthetic grid: behaviour is driven by projection at signal_layer."""
        rng = np.random.default_rng(seed)
        proj = rng.normal(size=(n, n_layers))
        behaviour = proj[:, signal_layer] + noise * rng.normal(size=n)
        return proj, behaviour

    def test_layerwise_recovers_signal_layer(self):
        proj, beh = self._grid(signal_layer=3, noise=0.05)
        curve = layerwise_correlation(proj, beh)
        rs = [c.r for c in curve]
        assert int(np.argmax(rs)) == 3          # signal layer wins
        assert rs[3] > 0.9                        # strong r there
        assert abs(rs[0]) < 0.6                    # noise layers weak

    def test_cv_is_honest_on_pure_noise(self):
        # no real signal → cross-validated r should be near 0, not inflated
        rng = np.random.default_rng(7)
        proj = rng.normal(size=(24, 8))
        beh = rng.normal(size=24)          # independent
        cv = cv_best_layer_correlation(proj, beh)
        assert abs(cv["cv_r"]) < 0.55       # not spuriously ~1

    def test_cv_recovers_real_signal(self):
        proj, beh = self._grid(n=30, signal_layer=2, noise=0.1)
        cv = cv_best_layer_correlation(proj, beh)
        assert cv["cv_r"] > 0.85
        assert cv["modal_layer"] == 2

    def test_summarise_shape(self):
        proj, pad = self._grid(signal_layer=1)
        _, vd = self._grid(signal_layer=4, seed=2)
        s = summarise_correlation(proj, pad, vd)
        assert s["n_cells"] == 20
        assert len(s["vs_pad"]["per_layer_r"]) == 6
        assert "cv" in s["vs_vd"]


# ── representation provider / extract / probe / steering (offline, mocked) ────

class _Cap:
    """Minimal stand-in for RepresentationProvider.capture's result."""
    def __init__(self, pooled):
        self.pooled = np.asarray(pooled, dtype=np.float64)
        self.provenance = {"model": "mock", "pooling": "response_avg", "chat_format": True}


class TestProviderPooling:
    def test_pool_capture_is_response_only(self):
        from personascope.repr.vllm_lens_provider import pool_capture
        # [n_layers=2, n_pos=5, hidden=2]; n_generated=2 → last 2 positions
        res = np.arange(2 * 5 * 2, dtype=float).reshape(2, 5, 2)
        pooled = pool_capture(res, n_generated=2, how="response_avg")
        assert pooled.shape == (2, 2)
        assert np.allclose(pooled, res[:, 3:, :].mean(axis=1))   # positions 3,4

    def test_pool_capture_clamps_overlong_ngen(self):
        from personascope.repr.vllm_lens_provider import pool_capture
        res = np.ones((2, 3, 2))
        # n_gen >= n_pos → prompt_len clamps to 0, pool all positions
        out = pool_capture(res, n_generated=99, how="response_avg")
        assert out.shape == (2, 2)


class TestSteeringControls:
    def test_random_control_matches_per_layer_norm(self):
        from personascope.probes.representation.steering_probe import random_control_direction
        d = np.random.default_rng(0).normal(size=(4, 8)) * 3.0
        rc = random_control_direction(d, seed=1)
        assert np.allclose(np.linalg.norm(rc, axis=-1), np.linalg.norm(d, axis=-1))
        # different orientation (not just a rescale of d)
        assert not np.allclose(rc, d)

    def test_opposite_direction(self):
        from personascope.probes.representation.steering_probe import (
            build_conditions, opposite_direction)
        d = np.random.default_rng(2).normal(size=(3, 5))
        assert np.allclose(opposite_direction(d), -d)
        conds = build_conditions(d, seed=0)
        assert set(conds) == {"direction", "random", "opposite"}


class TestExtractAndProbe:
    def _mock_capture(self, layer_signal):
        """capture(messages, max_tokens=...) -> _Cap. Pooled encodes the system
        polarity so pos/neg differ, at `layer_signal`."""
        n_layers, hidden = 4, 6
        def cap(messages, max_tokens=48):
            sys = next((m["content"] for m in messages if m["role"] == "system"), "")
            pol = 1.0 if "POS" in sys else (-1.0 if "NEG" in sys else 0.0)
            arr = np.zeros((n_layers, hidden))
            arr[layer_signal] = pol
            return _Cap(arr)
        return cap

    def test_extract_direction_recovers_contrast(self):
        from personascope.repr.extract import extract_direction
        cap = self._mock_capture(layer_signal=2)
        d, prov = extract_direction(cap, "POS trait", "NEG trait",
                                    ["q1", "q2", "q3"], max_tokens=8)
        assert d.shape == (4, 6)
        # signal layer separates (pos - neg = +2 at that dim), others ~0
        assert d[2].max() > 1.5 and abs(d[0].max()) < 1e-9
        assert prov["kind"] == "mean_diff_contrast"
        assert prov["n_extract_questions"] == 3 and "extract_questions_sha" in prov

    def test_project_cell_uses_direction(self):
        from personascope.probes.representation.persona_probe import project_cell
        # a cell whose activations align with a known direction at layer 2
        direction = np.zeros((4, 6)); direction[2, 0] = 1.0
        def cap(messages, max_tokens=48):
            a = np.zeros((4, 6)); a[2, 0] = 3.0
            return _Cap(a)
        out = project_cell(cap, direction, ["e1", "e2"], max_tokens=8)
        assert len(out["per_layer_mean"]) == 4
        assert abs(out["per_layer_mean"][2] - 3.0) < 1e-6   # proj at signal layer
        assert out["n_questions"] == 2
