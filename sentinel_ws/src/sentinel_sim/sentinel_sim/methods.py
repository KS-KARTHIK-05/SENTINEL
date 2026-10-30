"""Method registry: SENTINEL, its ablations and baselines B0-B8.

Each entry maps a name to (agent class, AgentConfig overrides, SpringConfig
overrides). All methods share sensing, gossip, ledger and channel code, so
differences come only from the mobility/coordination rule.
"""
from __future__ import annotations

from dataclasses import replace

from sentinel_core.agent import AgentConfig, SentinelAgent
from sentinel_core.baselines import (HardWallAgent, Lambda2Agent, LawnmowerAgent, MappoAgent,
                                     PotentialAgent, RelayChainAgent)
from sentinel_core.fitness import SpringConfig

# Baselines keep their published behaviour: constraints always on (if any), no
# heartbeat return, no information-triggered delivery.
_BASE = dict(connectivity='strict', reconnect=False)
_NEVER = 1e5                      # T_hold that never triggers within a mission
_NO_COMM = dict(neighbour_set='none', projection=False, sa_enabled=False, fusion=False, **_BASE)

METHODS = {
    # proposed: information-triggered intermittent connectivity (Amendment A1)
    'sentinel': (SentinelAgent, {}, {}),
    # previous operating point: always connected to the responder
    'sentinel_strict': (SentinelAgent, dict(connectivity='strict'), {}),
    # T_hold sweep (E10): delivery trigger age of unsecured detections
    'sentinel_th0': (SentinelAgent, dict(t_hold_s=0.0), {}),
    'sentinel_th30': (SentinelAgent, dict(t_hold_s=30.0), {}),
    'sentinel_th120': (SentinelAgent, dict(t_hold_s=120.0), {}),
    # baselines
    'B0_pso': (SentinelAgent, _NO_COMM, dict(mode='none')),
    'B1_v1': (SentinelAgent, dict(neighbour_set='nn', projection=False, sa_enabled=True,
                                  fusion=False, **_BASE),
              dict(mode='distance_nn', lam0=0.5, r_max=50.0, beta=0.0)),
    'B2_commpso': (SentinelAgent, dict(neighbour_set='all', projection=False, sa_enabled=False,
                                       fusion=False, **_BASE), dict(mode='outage', beta=0.0)),
    'B3_lambda2': (Lambda2Agent, _NO_COMM, dict(mode='none')),
    'B4_potential': (PotentialAgent, _NO_COMM, dict(mode='none')),
    'B5_mappo': (MappoAgent, _NO_COMM, dict(mode='none')),
    'B6_lawnmower': (LawnmowerAgent, _NO_COMM, dict(mode='none')),
    # baselines added for the ICIUS revision (overnight 2026-10-28): hard 50 m wall, relay chain
    'B7_wall': (HardWallAgent, _NO_COMM, dict(mode='none')),
    'B8_relay': (RelayChainAgent, _NO_COMM, dict(mode='none')),
    # ablations of SENTINEL (E7)
    'abl_distance_spring': (SentinelAgent, {}, dict(mode='distance_nn', lam0=0.5, beta=0.0)),
    'abl_nn_set': (SentinelAgent, dict(neighbour_set='nn'), {}),
    'abl_no_info_tension': (SentinelAgent, {}, dict(beta=0.0)),
    'abl_no_projection': (SentinelAgent, dict(projection=False), {}),
    'abl_no_sa': (SentinelAgent, dict(sa_enabled=False), {}),
    'abl_no_fusion': (SentinelAgent, dict(fusion=False), {}),
    'abl_no_sa_no_fusion': (SentinelAgent, dict(sa_enabled=False, fusion=False), {}),
    # previous fusion rule (plain naive-Bayes log-odds sum), ablation of Amendment A3
    'abl_naive_fusion': (SentinelAgent, dict(fusion_rule='naive_bayes'), {}),
    # attribution of the margin (E12): delivery and sensing components removed jointly
    'sentinel_th_inf': (SentinelAgent, dict(t_hold_s=_NEVER), {}),
    'abl_no_fusion_no_delivery': (SentinelAgent, dict(fusion=False, t_hold_s=_NEVER), {}),
    'abl_no_sa_no_fusion_no_delivery': (SentinelAgent, dict(sa_enabled=False, fusion=False,
                                                            t_hold_s=_NEVER), {}),
}

# Baselines with SENTINEL's corroborative fusion added to their sensing (E12): same mobility
# and tuned parameters, only the ledger keeps per-sector views and fuses them.
for _b in ['B0_pso', 'B1_v1', 'B2_commpso', 'B3_lambda2', 'B4_potential', 'B5_mappo',
           'B6_lawnmower', 'B7_wall', 'B8_relay']:
    _cls, _a, _s = METHODS[_b]
    METHODS[_b + '_fusion'] = (_cls, {**_a, 'fusion': True}, _s)

# Sensitivity of the spring parameters (overnight 2026-10-28, lambda, beta, epsilon and
# R_max): one parameter at a time around the tuned SENTINEL configuration. Each variant fixes
# only its own parameter; everything else inherits SENTINEL's tuned values (runner). epsilon is
# the outage target of both the spring (eps0) and the preserved-link radius R_eps (r_hard_eps).
for _v in (0.3, 4.8):
    METHODS[f'sentinel_lam0_{_v:g}'] = (SentinelAgent, {}, dict(lam0=_v))
for _v in (0.0, 4.0):
    METHODS[f'sentinel_beta_{_v:g}'] = (SentinelAgent, {}, dict(beta=_v))
for _v in (0.01, 0.2):
    METHODS[f'sentinel_eps_{_v:g}'] = (SentinelAgent, dict(r_hard_eps=_v), dict(eps0=_v))
for _r in (30, 50, 70):
    METHODS[f'abl_distance_spring_r{_r}'] = (SentinelAgent, {}, dict(
        mode='distance_nn', lam0=0.5, beta=0.0, r_max=float(_r)))


def build_config(method: str, base: AgentConfig | None = None, agent_overrides: dict | None = None,
                 spring_overrides: dict | None = None):
    cls, a_over, s_over = METHODS[method]
    base = base or AgentConfig()
    spring = replace(base.spring, **{**s_over, **(spring_overrides or {})})
    cfg = replace(base, spring=spring, **{**a_over, **(agent_overrides or {})})
    return cls, cfg
