import numpy as np
import pytest

from sentinel_core.channel import (LinkSampler, ChannelProfile, get_profile, outage_prob,
                                   reliable_range, PROFILES)


def test_rayleigh_closed_form():
    p = ChannelProfile('t', path_loss_exp=3.0, shadow_sigma_db=0.0, rician_k_db=float('-inf'))
    for d in (10.0, 50.0, 120.0):
        g = 10 ** (p.mean_snr_db(d) / 10)
        expected = 1 - np.exp(-(10 ** (p.gamma_th_db / 10)) / g)
        assert outage_prob(p, d) == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize('name', list(PROFILES))
def test_outage_monotone_in_distance_and_obstruction(name):
    p = PROFILES[name]
    d = np.linspace(5, 400, 50)
    po = outage_prob(p, d)
    assert np.all(np.diff(po) >= -1e-12)
    assert np.all(outage_prob(p, d, 1) >= po - 1e-12)


@pytest.mark.parametrize('name', ['suburban', 'urban_rubble', 'severe'])
def test_monte_carlo_matches_analytic(name):
    p = PROFILES[name]
    rng = np.random.default_rng(0)
    d = reliable_range(p, 0.2)
    lost = 0
    n = 20000
    for k in range(n):
        s = LinkSampler(p, rng)          # fresh sampler => independent shadowing
        lost += not s.delivered(0, 1, np.zeros(2), np.array([d, 0.0]), 0)
    assert lost / n == pytest.approx(0.2, abs=0.015)


def test_reliable_range_inverts_outage():
    p = get_profile('urban_rubble')
    for eps in (0.01, 0.05, 0.1):
        r = reliable_range(p, eps, 1)
        assert float(outage_prob(p, r, 1)) == pytest.approx(eps, abs=1e-4)


def test_urban_nlos_range_matches_v1_scale():
    # Calibration target: NLOS reliable range near the 50 m used by SENTINEL v1.
    assert 40.0 < reliable_range(get_profile('urban_rubble'), 0.05, 1) < 60.0


@pytest.mark.parametrize('name', list(PROFILES))
def test_outage_table_matches_exact(name):
    from sentinel_core.channel import outage_table
    p = PROFILES[name]
    tab = outage_table(p)
    d = np.exp(np.random.default_rng(0).uniform(np.log(2), np.log(3000), 500))
    for k in range(p.max_obstructions + 1):
        assert np.max(np.abs(tab(d, k) - outage_prob(p, d, k))) < 2e-3
