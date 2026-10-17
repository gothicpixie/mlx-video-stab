import numpy as np

from mlxstab import synth


def test_smooth_camera_intent_preserves_historical_formula():
    T = 24
    intended, _actual = synth.camera(T, seed=17, intent_mode="smooth")
    t = np.arange(T, dtype=np.float64)
    expected = np.stack((40 * np.sin(t / 70) + 0.5 * t,
                         10 * np.sin(t / 55), 1.2 * np.sin(t / 90)), axis=-1)
    np.testing.assert_array_equal(intended(t), expected)


def test_intent_modes_are_paired_with_identical_camera_shake():
    smooth_intent, smooth_actual = synth.camera(80, seed=23, jitter=6, rot_jit=0.6, intent_mode="smooth")
    random_intent, random_actual = synth.camera(80, seed=23, jitter=6, rot_jit=0.6, intent_mode="stochastic")
    tau = np.linspace(0, 79, 240)
    np.testing.assert_allclose(smooth_actual(tau) - smooth_intent(tau),
                               random_actual(tau) - random_intent(tau), atol=1e-12)
    assert not np.allclose(smooth_intent(tau), random_intent(tau))
    np.testing.assert_array_equal(random_intent(tau),
                                  synth.camera(80, seed=23, jitter=6, rot_jit=0.6,
                                               intent_mode="stochastic")[0](tau))


def test_matched_spectrum_changes_phase_without_changing_path_energy():
    T = 96
    frame = np.arange(T, dtype=np.float64)
    smooth, smooth_actual = synth.camera(T, seed=23, jitter=4, rot_jit=0.4,
                                         intent_mode="smooth", intent_profile="matched-spectrum")
    random, random_actual = synth.camera(T, seed=23, jitter=4, rot_jit=0.4,
                                         intent_mode="stochastic", intent_profile="matched-spectrum")
    smooth_path = smooth(frame)
    random_path = random(frame)
    smooth_path[:, 0] -= 0.5 * frame
    random_path[:, 0] -= 0.5 * frame

    smooth_fft = np.fft.rfft(smooth_path, axis=0)
    random_fft = np.fft.rfft(random_path, axis=0)
    np.testing.assert_allclose(np.abs(smooth_fft), np.abs(random_fft), atol=1e-10)
    smooth_cross = smooth_fft[:, :, None] * smooth_fft[:, None, :].conj()
    random_cross = random_fft[:, :, None] * random_fft[:, None, :].conj()
    np.testing.assert_allclose(smooth_cross, random_cross, atol=1e-8)
    omega = (2.0 * np.pi * np.fft.rfftfreq(T))[:, None]
    for order in (1, 2):
        smooth_d = np.fft.irfft((1j * omega) ** order * smooth_fft, n=T, axis=0)
        random_d = np.fft.irfft((1j * omega) ** order * random_fft, n=T, axis=0)
        np.testing.assert_allclose(np.sqrt(np.mean(smooth_d ** 2, axis=0)),
                                   np.sqrt(np.mean(random_d ** 2, axis=0)), atol=1e-10)
    assert not np.allclose(smooth_path, random_path)
    random_again, _ = synth.camera(T, seed=23, jitter=4, rot_jit=0.4,
                                   intent_mode="stochastic", intent_profile="matched-spectrum")
    np.testing.assert_array_equal(random(frame), random_again(frame))
    tau = np.linspace(0, T - 1, 260)
    np.testing.assert_allclose(smooth_actual(tau) - smooth(tau),
                               random_actual(tau) - random(tau), atol=1e-12)


def test_static_intent_is_zero_for_calibration_anchors():
    intended, actual = synth.camera(12, seed=4, jitter=0, rot_jit=0, intent_mode="static")
    t = np.arange(12, dtype=np.float64)
    np.testing.assert_array_equal(intended(t), np.zeros((12, 3)))
    np.testing.assert_array_equal(actual(t), np.zeros((12, 3)))


def test_matched_spectrum_rejects_too_short_windows():
    with np.testing.assert_raises_regex(ValueError, "at least 16 frames"):
        synth.camera(15, intent_profile="matched-spectrum")


def test_render_default_matches_explicit_historical_controls():
    args = dict(T=4, W=48, H=32, seed=5, rho=0.0, jitter=0.0, rot_jit=0.0)
    legacy, legacy_intent = synth.render(**args)
    explicit, explicit_intent = synth.render(**args, depths=(1.0, 0.55, 0.25), subject=True,
                                              subject_speed=1.6, intent_mode="smooth")
    np.testing.assert_array_equal(legacy, explicit)
    np.testing.assert_array_equal(legacy_intent, explicit_intent)


def test_render_validates_scene_factor_controls():
    with np.testing.assert_raises(ValueError):
        synth.render(T=2, W=32, H=24, depths=(1.0, 0.5))
    with np.testing.assert_raises(ValueError):
        synth.render(T=2, W=32, H=24, subject_speed=float("nan"))
    with np.testing.assert_raises(ValueError):
        synth.render(T=2, W=32, H=24, intent_mode="unknown")


def _yellow_subject_center(frame):
    mask = (frame[..., 0] > 180) & (frame[..., 1] > 160) & (frame[..., 2] < 100)
    y, x = np.nonzero(mask)
    assert len(x) > 20
    return np.array([x.mean(), y.mean()])


def test_subject_world_motion_is_independent_of_camera_motion():
    moving, _ = synth.render(T=8, W=128, H=96, seed=5, rho=0, jitter=0, rot_jit=0,
                            subject_speed=2.0, intent_mode="static")
    centers = np.stack([_yellow_subject_center(moving[t]) for t in (0, 7)])
    np.testing.assert_allclose(centers[1, 0] - centers[0, 0], 14.0, atol=0.35)

    static_subject, _ = synth.render(T=8, W=128, H=96, seed=5, rho=0, jitter=0, rot_jit=0,
                                    subject_speed=0.0, intent_mode="smooth")
    pose, _actual = synth.camera(8, seed=6, jitter=0, rot_jit=0, intent_mode="smooth")
    pose_delta = pose(np.array([7.0]))[0, 0] - pose(np.array([0.0]))[0, 0]
    centers = np.stack([_yellow_subject_center(static_subject[t]) for t in (0, 7)])
    np.testing.assert_allclose(centers[1, 0] - centers[0, 0], -pose_delta, atol=0.4)


def test_truth_visibility_contains_far_mid_and_near_layers():
    _frames, _intent, truth = synth.render(T=3, W=160, H=96, seed=19, rho=0, jitter=0, rot_jit=0,
                                          depths=(1.0, 1.0, 1.0), subject=False,
                                          intent_mode="static", return_truth=True)
    visible = truth["visible_layer_id"]
    assert visible.shape == (3, 96, 160)
    assert set(np.unique(visible)).issuperset({0, 1, 2})
