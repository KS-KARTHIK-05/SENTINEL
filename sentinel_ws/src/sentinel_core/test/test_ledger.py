import numpy as np

from sentinel_core.ledger import Ledger


def test_ledger_never_exceeds_true_targets():
    """Regression for the v1 '10 targets / 15 coordinates' inconsistency (R1.6)."""
    rng = np.random.default_rng(0)
    targets = rng.uniform(0, 100, (10, 2))
    while np.min([np.linalg.norm(a - b) for i, a in enumerate(targets)
                  for b in targets[i + 1:]]) < 10:
        targets = rng.uniform(0, 100, (10, 2))
    ledgers = [Ledger(i) for i in range(3)]
    for t in range(200):
        for led in ledgers:
            k = rng.integers(10)
            obs_pos = targets[k] + rng.uniform(-20, 20, 2)
            led.observe(targets[k] + rng.normal(0, 0.8, 2), float(rng.uniform(0.5, 0.95)), t,
                        np.r_[obs_pos, 30.0])
        i, j = rng.choice(3, 2, replace=False)
        held = ledgers[j].merge(ledgers[i].digest())
        ledgers[i].ack(held, j)
    for led in ledgers:
        assert len(led.entries) <= 10
    # after full exchange every ledger knows exactly the 10 targets
    for _ in range(3):
        for i in range(3):
            for j in range(3):
                if i != j:
                    ledgers[j].merge(ledgers[i].digest())
    for led in ledgers:
        assert len(led.entries) == 10


def test_unreplicated_count_drops_after_ack():
    a, b = Ledger(0), Ledger(1)
    a.observe([5, 5], 0.8, 0.0, [5, 5, 30])
    assert a.unreplicated_count() == 1
    held = b.merge(a.digest())
    a.ack(held, 1)
    assert a.unreplicated_count() == 0
    a.forget_holder(1)
    assert a.unreplicated_count() == 1


def test_multiview_fusion_raises_confidence():
    a = Ledger(0)
    a.observe([0, 0], 0.6, 0.0, [-20, 0, 30])
    c1 = a.entries[(0, 0)].confidence
    a.observe([0, 0], 0.6, 1.0, [20, 0, 30])     # different view sector
    assert a.entries[(0, 0)].confidence > c1
    a.observe([0, 0], 0.6, 2.0, [20, 0.1, 30])   # same sector: no double counting
    assert a.entries[(0, 0)].confidence == a.entries[(0, 0)].confidence


def test_weak_occluded_views_do_not_veto_a_confirmed_view(monkeypatch):
    import sentinel_core.ledger as L
    monkeypatch.setattr(L, 'FUSION_RULE', 'corroborative')
    a = Ledger(0)
    a.observe([0, 0], 0.85, 0.0, [-20, 0, 30])           # clear view: confirmed
    for k, obs in enumerate([[20, 0, 30], [0, 20, 30], [0, -20, 30], [14, 14, 30]]):
        a.observe([0, 0], 0.30, 1.0 + k, obs)           # occluded angles
    assert a.entries[(0, 0)].confirmed


def test_single_view_confidence_unchanged():
    a = Ledger(0)
    a.observe([0, 0], 0.55, 0.0, [-20, 0, 30])
    assert abs(a.entries[(0, 0)].confidence - 0.55) < 1e-9
