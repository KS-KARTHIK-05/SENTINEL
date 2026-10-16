"""Method registry: SENTINEL, its ablations and baselines B0-B4, B6.

Each entry maps a name to (agent class, AgentConfig overrides, SpringConfig
overrides). All methods share sensing, gossip, ledger and channel code, so
differences come only from the mobility/coordination rule.
"""
from __future__ import annotations

from dataclasses import replace

from sentinel_core.agent import AgentConfig, SentinelAgent
from sentinel_core.baselines import Lambda2Agent, LawnmowerAgent, MappoAgent, PotentialAgent
from sentinel_core.fitness import SpringConfig

# Baselines keep their published behaviour: constraints always on (if any), no
# heartbeat return, no information-triggered delivery.
_BASE = dict(connectivity='strict', reconnect=False)
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
}


def build_config(method: str, base: AgentConfig | None = None, agent_overrides: dict | None = None,
                 spring_overrides: dict | None = None):
    cls, a_over, s_over = METHODS[method]
    base = base or AgentConfig()
    spring = replace(base.spring, **{**s_over, **(spring_overrides or {})})
    cfg = replace(base, spring=spring, **{**a_over, **(agent_overrides or {})})
    return cls, cfg
