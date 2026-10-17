import json

import pytest

from bench import synth_intent_factorial as factorial


def test_design_has_predeclared_2x2x3_cells_and_three_calibration_anchors():
    design = factorial.build_design(seeds=(17, 23), frames=96, width=320, height=192)
    cells = design["cells_per_seed"]
    main = [cell for cell in cells if cell["kind"] == "main"]
    anchors = [cell for cell in cells if cell["kind"] == "anchor"]
    assert len(main) == 12
    assert len({(c["intent_mode"], c["layer_mode"], c["subject_mode"]) for c in main}) == 12
    assert len(anchors) == 3
    assert {c["id"] for c in anchors} == {"intent_only", "shake_only", "neither"}
    assert design["seeds"] == [17, 23]
    assert all(c["layer_motion_factors"][0] == 1.0 for c in cells)


def test_design_uses_common_random_numbers_for_intent_and_shake_cells():
    cells = factorial.build_cells()
    smooth = [c for c in cells if c["kind"] == "main" and c["intent_mode"] == "smooth"]
    stochastic = [c for c in cells if c["kind"] == "main" and c["intent_mode"] == "stochastic"]
    assert len(smooth) == len(stochastic) == 6
    assert {(c["layer_mode"], c["subject_mode"], c["jitter_px"], c["rot_jitter_deg"]) for c in smooth} == {
        (c["layer_mode"], c["subject_mode"], c["jitter_px"], c["rot_jitter_deg"]) for c in stochastic
    }
    assert all(c["jitter_px"] == 3.0 for c in smooth + stochastic)


def test_design_rejects_unpowered_or_duplicate_inputs():
    with pytest.raises(ValueError, match=">=64 frames"):
        factorial.build_design(frames=32)
    with pytest.raises(ValueError, match="unique integers"):
        factorial.build_design(seeds=(1, 1))


def test_manifest_resume_requires_matching_scene_configuration():
    design = factorial.build_design(seeds=(3,))
    cell = design["cells_per_seed"][0]
    expected = factorial._expected_synthetic(3, cell, design)
    manifest = {"status": "success", "source": {"synthetic": expected},
                "outputs": {"plan": "plan_r0.npz"}}
    assert factorial._manifest_matches(manifest, expected)
    changed = dict(manifest)
    changed["source"] = {"synthetic": {**expected, "jitter_px": 4.0}}
    assert not factorial._manifest_matches(changed, expected)


def test_limited_screen_records_pending_cells_without_inventing_results(tmp_path, monkeypatch):
    design = factorial.build_design(seeds=(1, 2))
    def fake_run_one(seed, cell, design, out_root, resume=True):
        return {"seed": seed, "cell_id": cell["id"], "kind": cell["kind"], "status": "success"}
    monkeypatch.setattr(factorial, "_run_one", fake_run_one)
    result = factorial.run_screen(design, tmp_path, limit_cells=1, log=lambda _: None)
    assert result["counts"] == {"planned": 30, "recorded": 2, "succeeded": 2, "failed": 0, "pending": 28}
    disk = json.loads((tmp_path / "screen.json").read_text())
    assert len(disk["runs"]) == 2
