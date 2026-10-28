"""Tests for the overnight baselines (B7 hard wall, B8 relay chain) and the sensitivity variants."""
from __future__ import annotations

import numpy as np
import pytest

from sentinel_core.agent import AgentConfig, NeighbourState, SentinelAgent
from sentinel_core.baselines.hard_wall import HardWallAgent, nearest_distance
from sentinel_core.baselines.relay_chain import RelayChainAgent, chain_targets, n_relays
from sentinel_core.channel import get_profile, reliable_range
from sentinel_core.maps import SearchMap
from sentinel_sim.methods import METHODS, build_config
from sentinel_sim.runner import expand
from sentinel_sim.sim import Simulation

SIZE = 200.0
TUNED = {'sentinel': {'agent': {'t_hold_s': 116.0, 'w_goal': 0.07},
                      'spring': {'lam0': 1.2, 'beta': 1.09}},
         'B7_wall': {'agent': {'w_goal': 0.5}},
         'B8_relay': {'agent_attrs': {'rho': 0.3, 'spacing': 0.7}}}


def _agent(cls, method, uid=0, pos=(100.0, 100.0), seed=0):
    _, cfg = build_config(method, AgentConfig())
    bounds = (np.zeros(2), np.full(2, SIZE))
    return cls(uid, pos, cfg, get_profile('urban_rubble'), SearchMap((SIZE, SIZE), 2.0, 4),
               bounds, np.random.default_rng(seed))


def _nbr(x, y):
    p = np.array([x, y], float)
    return NeighbourState(p, p.copy(), 0.0, 5.0)


# ------------------------------------------------------------------ B7 hard wall
def test_nearest_distance():
    d = nearest_distance(np.array([[0.0, 0.0], [10.0, 0.0]]), np.array([[3.0, 4.0], [10.0, 1.0]]))
    assert np.allclose(d, [5.0, 1.0])


def test_wall_rejects_candidates_beyond_radius():
    a = _agent(HardWallAgent, 'B7_wall')
    nbrs = {1: _nbr(130.0, 100.0)}
    f, seeds = a._search_hook(0.0, lambda p: np.ones(len(np.atleast_2d(p))), [], nbrs)
    vals = f(np.array([[150.0, 100.0], [190.0, 100.0], [100.0, 100.0]]))
    assert vals[0] == 1.0 and vals[2] == 1.0 and vals[1] == -np.inf
    assert len(seeds) == 1          # approach point toward the nearest neighbour


def test_wall_waypoints_stay_within_radius():
    a = _agent(HardWallAgent, 'B7_wall')
    nbrs = {1: _nbr(140.0, 100.0)}
    a.nbrs = nbrs
    for k in range(5):
        a._plan(float(k), nbrs)
        assert nearest_distance(a.waypoint, np.array([[140.0, 100.0]]))[0] <= a.r_wall + 1e-9


def test_wall_holds_when_every_candidate_is_infeasible():
    a = _agent(HardWallAgent, 'B7_wall')
    nbrs = {1: _nbr(190.0, 190.0)}          # 127 m away, beyond wall + search radius (20 m)
    a.nbrs = nbrs
    a._plan(0.0, nbrs)
    assert np.allclose(a.waypoint, a.pos)


def test_wall_without_neighbours_equals_b0():
    a = _agent(HardWallAgent, 'B7_wall', seed=3)
    b = _agent(SentinelAgent, 'B0_pso', seed=3)
    a._plan(0.0, {})
    b._plan(0.0, {})
    assert np.allclose(a.waypoint, b.waypoint)


# ------------------------------------------------------------------ B8 relay chain
@pytest.mark.parametrize('n,rho,m', [(8, 0.25, 2), (8, 0.0, 1), (3, 0.0, 1), (2, 0.5, 1),
                                     (4, 0.5, 2), (1, 0.5, 0), (50, 0.5, 25), (3, 0.5, 2)])
def test_relay_count(n, rho, m):
    assert n_relays(n, rho) == m


def test_chain_targets_even_spacing_and_cap():
    t = chain_targets(np.zeros(2), np.array([300.0, 0.0]), 3, 40.0)
    assert np.allclose(t, [[40, 0], [80, 0], [120, 0]])
    t = chain_targets(np.zeros(2), np.array([60.0, 0.0]), 3, 40.0)     # centroid near: D / m
    assert np.allclose(t, [[20, 0], [40, 0], [60, 0]])


def _sim(n=8, extra=None):
    cfg = {'method': 'B8_relay', 'seed': 1, 'n_uav': n, 'duration_s': 20.0,
           'map': {'kind': 'generate', 'size': 200, 'n_targets': 5, 'base': True},
           'agent_attrs': {'rho': 0.25, 'spacing': 0.8}, **(extra or {})}
    return Simulation(cfg)


def test_relay_roles_and_sectors():
    s = _sim()
    relays = [a.id for a in s.agents[:s.N] if a.is_relay]
    assert relays == [6, 7]
    assert all(a.cfg.n_agents == 6 for a in s.agents[:s.N] if not a.is_relay)


def test_relay_chain_reforms_after_failure():
    s = _sim()
    r6, r7 = s.agents[6], s.agents[7]
    gap = 0.8 * reliable_range(s.profile, 0.05, 1)
    for a in s.agents[:6]:
        a.pos = np.array([150.0, 150.0])
    base = np.asarray(s.world.base_xy, float)
    v = np.array([150.0, 150.0]) - base
    u = v / np.linalg.norm(v)
    step = min(gap, np.linalg.norm(v) / 2)
    assert np.allclose(r6._chain_target(), base + u * step)
    assert np.allclose(r7._chain_target(), base + 2 * u * step)
    r6.alive = False
    step = min(gap, np.linalg.norm(v))
    assert np.allclose(r7._chain_target(), base + u * step)


def test_relays_do_not_search_and_mission_runs():
    s = _sim()
    res = s.run()['summary']
    assert 0.0 <= res['tdr'] <= 1.0
    r = s.agents[7]
    assert r.goal is None                  # never ran the search planner


def test_hard_wall_mission_runs():
    res = Simulation({'method': 'B7_wall', 'seed': 2, 'n_uav': 4, 'duration_s': 20.0,
                      'map': {'kind': 'generate', 'size': 200, 'n_targets': 5,
                              'base': True}}).run()['summary']
    assert 0.0 <= res['tdr'] <= 1.0


# ------------------------------------------------------------------ registry and variants
def test_new_fusion_variants_registered():
    for b in ('B7_wall', 'B8_relay'):
        cls, a, s = METHODS[b + '_fusion']
        assert cls is METHODS[b][0] and a['fusion'] is True and s == METHODS[b][2]


@pytest.mark.parametrize('method,key,section,value', [
    ('sentinel_lam0_0.3', 'lam0', 'spring', 0.3), ('sentinel_lam0_4.8', 'lam0', 'spring', 4.8),
    ('sentinel_beta_0', 'beta', 'spring', 0.0), ('sentinel_beta_4', 'beta', 'spring', 4.0),
    ('sentinel_eps_0.01', 'eps0', 'spring', 0.01), ('sentinel_eps_0.2', 'eps0', 'spring', 0.2),
    ('abl_distance_spring_r30', 'r_max', 'spring', 30.0),
    ('abl_distance_spring_r70', 'r_max', 'spring', 70.0)])
def test_sensitivity_variant_changes_only_its_parameter(method, key, section, value):
    spec = {'base': {}, 'method_overrides': TUNED, 'methods': [method], 'seeds': 1}
    cfg = next(expand(spec))
    _, ac = build_config(method, AgentConfig(), cfg.get('agent'), cfg.get('spring'))
    _, ref = build_config('sentinel', AgentConfig(), TUNED['sentinel']['agent'],
                          TUNED['sentinel']['spring'])
    assert getattr(ac.spring, key) == value
    assert ac.t_hold_s == ref.t_hold_s and ac.w_goal == ref.w_goal   # tuned values inherited
    if not method.startswith('abl_'):
        for k in ('lam0', 'beta', 'eps0', 'mode'):
            if k != key:
                assert getattr(ac.spring, k) == getattr(ref.spring, k)


def test_eps_variant_sets_both_outage_targets():
    _, ac = build_config('sentinel_eps_0.2', AgentConfig())
    assert ac.r_hard_eps == 0.2 and ac.spring.eps0 == 0.2


def test_tuned_values_reach_new_baselines():
    spec = {'base': {}, 'method_overrides': TUNED, 'methods': ['B8_relay_fusion'], 'seeds': 1}
    cfg = next(expand(spec))
    assert cfg['agent_attrs'] == {'rho': 0.3, 'spacing': 0.7}
