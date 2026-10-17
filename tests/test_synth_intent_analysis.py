from bench.synth_intent_analysis import analyze


def _run(seed, mode, index, *, intent_rmse, shake_db, status="success"):
    subject = ("absent", "static", "moving")[index % 3]
    return {
        "seed": seed,
        "kind": "main",
        "cell_id": f"{mode}__{'far_only' if index < 3 else 'parallax'}__{subject}",
        "status": status,
        "far_camera_metrics": {
            "intent_retention": {
                "translation_x_px": {"rmse": intent_rmse},
                "translation_y_px": {"rmse": intent_rmse + 1},
                "rotation_deg": {"rmse": intent_rmse + 2},
            },
            "shake_rejection": {
                "translation_x_px": {"attenuation_db": shake_db},
                "translation_y_px": {"attenuation_db": shake_db - 1},
                "rotation_deg": {"attenuation_db": shake_db - 2},
            },
        },
    }


def test_analyze_uses_seed_as_unit_and_excludes_calibration_anchors():
    runs = []
    for seed, smooth, stochastic in ((4, 3.0, 2.0), (8, 1.0, 4.0)):
        for index in range(6):
            runs.append(_run(seed, "smooth", index, intent_rmse=smooth, shake_db=-10))
            runs.append(_run(seed, "stochastic", index, intent_rmse=stochastic, shake_db=-15))
    cell = next(run for run in runs if run["seed"] == 4 and run["cell_id"] == "stochastic__far_only__absent")
    cell["far_camera_metrics"]["intent_retention"]["translation_x_px"]["rmse"] += 3.0
    runs.append({"seed": 4, "kind": "anchor", "cell_id": "shake_only", "status": "success"})

    report = analyze({"runs": runs})
    result = report["summaries"]["intent_x_rmse_px"]
    assert report["experimental_unit"] == "seed"
    assert report["complete_seeds"] == [4, 8]
    assert result["n_seeds"] == 2
    assert [row["delta_stochastic_minus_smooth"] for row in result["per_seed"]] == [-0.5, 3.0]
    assert result["mean_delta_stochastic_minus_smooth"] == 1.25
    assert result["negative_seed_deltas"] == 1
    assert result["positive_seed_deltas"] == 1
    assert result["stratified"]["by_layer"]["far_only"]["n_seeds"] == 2
    assert result["stratified"]["by_subject"]["moving"]["n_seeds"] == 2
    assert result["stratified"]["by_cell"]["far_only__absent"]["n_seeds"] == 2
    layer_interaction = result["factorial_interactions"]["phase_by_layer"]
    assert [row["delta_stochastic_minus_smooth"] for row in layer_interaction["per_seed"]] == [-1.0, 0.0]
    shake = report["summaries"]["shake_x_residual_db"]
    assert [row["delta_stochastic_minus_smooth"] for row in shake["per_seed"]] == [-5.0, -5.0]
    assert "negative values mean attenuation" in report["interpretation"]["shake_residual_db"]


def test_analyze_omits_seed_with_failed_cell():
    runs = []
    for index in range(6):
        runs.append(_run(3, "smooth", index, intent_rmse=2.0, shake_db=-10))
        runs.append(_run(3, "stochastic", index, intent_rmse=1.0, shake_db=-12,
                         status="failed" if index == 5 else "success"))

    report = analyze({"runs": runs})
    assert report["complete_seeds"] == []
    assert report["incomplete_seeds"] == [{
        "seed": 3, "smooth_cells": 6, "stochastic_cells": 6, "reason": "missing_or_failed_main_cells"
    }]
    assert report["summaries"]["intent_x_rmse_px"]["n_seeds"] == 0


def test_analyze_omits_seed_with_duplicate_cell_even_when_count_is_six():
    runs = []
    for index in range(6):
        runs.append(_run(5, "smooth", index, intent_rmse=2.0, shake_db=-10))
        runs.append(_run(5, "stochastic", index, intent_rmse=1.0, shake_db=-12))
    smooth_rows = [run for run in runs if run["seed"] == 5 and run["cell_id"].startswith("smooth__")]
    smooth_rows[-1]["cell_id"] = smooth_rows[0]["cell_id"]

    report = analyze({"runs": runs})
    assert report["complete_seeds"] == []
    assert report["incomplete_seeds"] == [{
        "seed": 5, "smooth_cells": 6, "stochastic_cells": 6, "reason": "main_cell_design_mismatch"
    }]


def test_analyze_reports_configured_seed_with_no_main_cells():
    report = analyze({"design": {"seeds": [5, 6]}, "runs": []})
    assert report["complete_seeds"] == []
    assert report["incomplete_seeds"] == [
        {"seed": 5, "smooth_cells": 0, "stochastic_cells": 0, "reason": "no_main_cells"},
        {"seed": 6, "smooth_cells": 0, "stochastic_cells": 0, "reason": "no_main_cells"},
    ]
