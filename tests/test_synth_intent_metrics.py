import numpy as np

from mlxstab.stabilize import Plan
from bench.synth_intent_metrics import (
    camera_to_world,
    fit_effective_camera_pose,
    recover_layer_pose_series,
    summarize_far_camera_path,
)
from mlxstab import synth


def _rotation(degrees):
    theta = np.deg2rad(degrees)
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def _identity_plan(frames, width, height):
    corners = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float64)
    return Plan(
        G=[np.eye(3, dtype=np.float64) for _ in range(frames)],
        mesh_corr=np.zeros((frames, 1, 1, 2), np.float64),
        row_corr=np.zeros((frames, 1, 2), np.float64),
        raw_quads=np.tile(corners, (frames, 1, 1)),
        path_quads=np.tile(corners, (frames, 1, 1)),
    )


def test_camera_to_world_and_identity_plan_recover_known_camera_pose():
    width, height, frames = 64, 48, 4
    center = np.array([(width - 1) / 2, (height - 1) / 2])
    visible = np.zeros((frames, height, width), np.uint8)
    plan = _identity_plan(frames, width, height)
    poses = np.array([[3.0 + t, -2.0, 1.5 * t] for t in range(frames)])

    def actual(tau):
        tau = np.asarray(tau)
        return np.column_stack((3.0 + tau, np.full(tau.shape, -2.0), 1.5 * tau))

    result = recover_layer_pose_series(plan, actual_pose_fn=actual, visible_layer_id=visible,
                                       width=width, height=height, rho=0, sample_step=8)
    assert all(row["pose"] is not None for row in result)
    for t, row in enumerate(result):
        pose = row["pose"]
        np.testing.assert_allclose(pose["translation_px"], poses[t, :2], atol=2e-5)
        np.testing.assert_allclose(pose["rotation_deg"], poses[t, 2], atol=2e-5)
        np.testing.assert_allclose(pose["scale"], 1.0, atol=1e-8)
        assert pose["fit_rmse_px"] < 2e-5


def test_composed_backward_warp_recovers_world_pose_in_correct_direction():
    width, height = 96, 72
    center = np.array([(width - 1) / 2, (height - 1) / 2])
    yy, xx = np.mgrid[0:height:8, 0:width:8]
    output = np.column_stack((xx.ravel(), yy.ravel())).astype(np.float64)
    camera_rotation, warp_rotation = 17.0, -6.0
    warp_translation = np.array([4.0, -3.0])
    input_xy = (output - center) @ _rotation(warp_rotation).T + center + warp_translation
    input_pose = np.array([7.0, -2.0, camera_rotation])
    world = camera_to_world(input_xy, input_pose, center)
    recovered = fit_effective_camera_pose(output, world, center, robust=False)

    expected_rotation = _rotation(camera_rotation) @ _rotation(warp_rotation)
    np.testing.assert_allclose(recovered["rotation_deg"], camera_rotation + warp_rotation, atol=1e-10)
    np.testing.assert_allclose(recovered["translation_px"],
                               input_pose[:2] + _rotation(camera_rotation) @ warp_translation, atol=1e-10)
    np.testing.assert_allclose(recovered["scale"], 1.0, atol=1e-10)
    np.testing.assert_allclose(_rotation(recovered["rotation_deg"]), expected_rotation, atol=1e-10)
    assert recovered["all_support_rmse_px"] < 1e-10


def test_layer_translation_factor_and_subject_world_offset_are_explicit():
    p = np.array([[10.0, 20.0], [30.0, 20.0], [10.0, 40.0]])
    center = np.array([20.0, 30.0])
    pose = np.array([8.0, -4.0, 0.0])
    near = camera_to_world(p, pose, center, translation_factor=1.5)
    subject_local = camera_to_world(p, pose, center, translation_factor=1.0, world_offset=(12.0, 0.0))
    np.testing.assert_allclose(near - subject_local, np.array([[16.0, -2.0]] * 3))


def test_path_summary_keeps_intent_and_shake_bands_disjoint():
    T = 96
    intended_fn, actual_fn = synth.camera(T, seed=11, jitter=3, rot_jit=0.2,
                                          intent_mode="smooth", intent_profile="matched-spectrum")
    t = np.arange(T, dtype=np.float64)
    intended, actual = intended_fn(t), actual_fn(t)
    rows = [{"frame": i, "valid_fraction": 1.0, "visible_fraction": 0.5,
             "pose": {"translation_px": intended[i, :2].tolist(),
                      "rotation_deg": float(intended[i, 2]), "fit_rmse_px": 0.0}}
            for i in range(T)]
    summary = summarize_far_camera_path(rows, intended, actual, matched_profile=True)
    assert summary["complete_case"] is True
    assert summary["frequency_bands_disjoint"] is True
    assert summary["intent_retention"]["translation_x_px"]["gain"] > 0.99
    assert abs(summary["intent_retention"]["shared_x_drift"]["gain"] - 1.0) < 1e-8
    for axis in summary["shake_rejection"].values():
        assert axis["output_residual_band_rms"] == 0.0
