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
