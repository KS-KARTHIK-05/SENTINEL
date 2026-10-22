import numpy as np

from sentinel_core.pso import PSOConfig, optimise


def test_pso_finds_peak_inside_disk():
    peak = np.array([12.0, -7.0])
    f = lambda p: -np.sum((p - peak) ** 2, axis=1)  # noqa: E731
    rng = np.random.default_rng(1)
    best, _ = optimise(f, np.zeros(2), 20.0, (np.full(2, -100.0), np.full(2, 100.0)), rng,
                       PSOConfig(particles=24, iterations=30))
    assert np.linalg.norm(best - peak) < 0.5


def test_pso_respects_disk_and_bounds():
    f = lambda p: p[:, 0]  # noqa: E731  maximise x -> boundary
    rng = np.random.default_rng(2)
    best, _ = optimise(f, np.zeros(2), 10.0, (np.full(2, -100.0), np.array([5.0, 100.0])), rng)
    assert best[0] <= 5.0 + 1e-9 and np.linalg.norm(best) <= 10.0 + 1e-9
