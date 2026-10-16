import numpy as np

from sentinel_core.maps import SearchMap


def test_gcounter_merge_is_idempotent_and_commutative():
    a = SearchMap((50, 50), 2.0, 2)
    b = SearchMap((50, 50), 2.0, 2)
    a.record_look(0, (10, 10), 6)
    b.record_look(1, (20, 20), 6)
    b.record_look(1, (20, 20), 6)
    r1 = a.residual().copy()
    a.merge(b)
    once = a.residual().copy()
    a.merge(b)
    assert np.array_equal(once, a.residual())
    assert np.all(once <= r1 + 1e-15)
    b.merge(a)
    assert np.array_equal(a.looks, b.looks)


def test_residual_follows_bayes_update():
    m = SearchMap((10, 10), 1.0, 1, p0=0.1, p_detect=0.8)
    m.record_look(0, (5, 5), 20)
    p = 0.1 * 0.2 / (0.1 * 0.2 + 0.9)
    assert np.allclose(m.residual(), p)


def test_versioned_merge_matches_full_max_and_total():
    rng = np.random.default_rng(0)
    maps = [SearchMap((60, 60), 2.0, 3) for _ in range(3)]
    for _ in range(50):
        k = int(rng.integers(3))
        maps[k].record_look(k, rng.uniform(0, 60, 2), 6)
        a, b = rng.choice(3, 2, replace=False)
        maps[b].merge_rows(*maps[a].rows())
    for m in maps:
        assert np.array_equal(m.total, m.looks.sum(0))
    for _ in range(3):
        for a in range(3):
            for b in range(3):
                maps[b].merge_rows(*maps[a].rows())
    assert all(np.array_equal(maps[0].looks, m.looks) for m in maps)
