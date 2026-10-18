"""Config, CLI and analyze() plumbing for the experimental mesh shape regularizer.

The low-level ``fit_motion`` keyword is ``shape_strength`` (default ``0``). The public
``Config`` field and CLI flag are ``mesh_shape_strength`` / ``--mesh-shape-strength``;
there are no aliases at either layer. The default of zero disables the regularizer.
"""
import numpy as np
import pytest
import mlx.core as mx

from mlxstab import motion as MO, stabilize as S


# --------------------------------------------------------------------------------------
# Config validation
# --------------------------------------------------------------------------------------

def test_config_mesh_shape_strength_defaults_to_disabled():
    cfg = S.Config()
    assert cfg.mesh_shape_strength == 0.0
    assert isinstance(cfg.mesh_shape_strength, float)
    assert S.Config(mesh_shape_strength=2).mesh_shape_strength == 2.0


@pytest.mark.parametrize("value", [0, 0.0, 1, 2.5, np.float32(0.25), np.float64(3.0)])
def test_config_accepts_finite_non_negative_mesh_shape_strength(value):
    assert S.Config(mesh_shape_strength=value).mesh_shape_strength == float(value)


@pytest.mark.parametrize("bad", [-1.0, -1e-9, np.nan, np.inf, -np.inf, True, False, np.bool_(True),
                                 "1", b"1", [1.0], np.array(1.0), 1 + 2j, None])
def test_config_rejects_invalid_mesh_shape_strength(bad):
    with pytest.raises(ValueError, match="mesh_shape_strength"):
        S.Config(mesh_shape_strength=bad)


# --------------------------------------------------------------------------------------
# CLI default, wiring and help text
# --------------------------------------------------------------------------------------

def _capture_main_config(monkeypatch, argv):
    import mlxstab.__main__ as M
    captured = {}

    def fake_stabilize(*args, **kwargs):
        captured["cfg"] = args[4]
        return {"ok": True}

    monkeypatch.setattr(M, "stabilize", fake_stabilize)
    M.main(argv)
    return captured["cfg"]


def test_main_defaults_mesh_shape_strength_to_disabled(monkeypatch):
    cfg = _capture_main_config(monkeypatch, ["in.mp4", "out.mp4"])
    assert cfg.mesh_shape_strength == 0.0


def test_main_wires_mesh_shape_strength(monkeypatch):
    cfg = _capture_main_config(monkeypatch, ["in.mp4", "out.mp4", "--mesh-shape-strength", "2.5"])
    assert cfg.mesh_shape_strength == 2.5


def test_main_help_marks_mesh_shape_strength_experimental_and_disabled(capsys):
    import mlxstab.__main__ as M
    with pytest.raises(SystemExit) as exc:
        M.main(["--help"])
    assert exc.value.code == 0
    text = " ".join(capsys.readouterr().out.split()).lower()
    assert "--mesh-shape-strength" in text
    assert "experimental" in text
    assert "disabled by default" in text


# --------------------------------------------------------------------------------------
# analyze() pass-through
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("strength", [0.0, 1.5])
def test_analyze_passes_mesh_shape_strength_into_fit_motion(monkeypatch, strength):
    W, H = 32, 24
    calls = []

    def fake_fit(obs, mesh, rs_rows, **kwargs):
        calls.append((mesh, rs_rows, kwargs))
        gc, gr = mesh
        return MO.Fit(np.eye(3), np.zeros((gr + 1, gc + 1, 2)), np.zeros((max(rs_rows, 1), 2)),
                      1.0, 0.0, False)

    def zero_flow(a, b, key):
        return mx.zeros((H, W, 2), dtype=mx.float32)

    monkeypatch.setattr(MO, "fit_motion", fake_fit)
    cfg = S.Config(mesh=(4, 3), rs_rows=2, mesh_shape_strength=strength)
    mot = S.analyze(iter([np.zeros((H, W, 3), np.uint8)] * 2), zero_flow, cfg, W, H, 30.0, threads=False)

    assert len(calls) == 1
    mesh, rs_rows, kwargs = calls[0]
    assert mesh == (4, 3) and rs_rows == 2
    assert kwargs["shape_strength"] == strength
    assert "mesh_shape_strength" not in kwargs
    assert mot.fits[0].fallback is False
