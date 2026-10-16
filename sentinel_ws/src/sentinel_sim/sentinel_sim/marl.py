"""MAPPO training for baseline B5 on the fast simulator.

Parameter-shared actor (local observations, decentralised execution) and a
centralised critic over [global state, local observation] (Yu et al. 2022).
Team reward per second: increase of the time-discounted report value plus a
small coverage term, i.e. the same objective the other methods are tuned on.
Training seeds (2000+) are disjoint from tuning (1000+) and evaluation (0-99).

    python -m sentinel_sim.marl --iters 120 --episodes 16 --out configs/mappo_policy.npz
"""
from __future__ import annotations

import os as _os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    _os.environ.setdefault(_v, '1')     # one BLAS thread per worker process (set before numpy)

import argparse
import copy
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

N_TRAIN = 8
SCENARIOS = [{'channel': {'profile': 'urban_rubble'}}, {'channel': {'profile': 'severe'}}]
BASE = {'method': 'B5_mappo', 'n_uav': N_TRAIN, 'duration_s': 600,
        'map': {'kind': 'generate', 'size': 400, 'n_targets': 25, 'base': True},
        'failures': {'kind': 'poisson', 'rate': 0.01}}
GLOBAL_DIM = 4 * N_TRAIN + 3


def rollout(args):
    actor, seed, scenario = args
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    from sentinel_core.baselines.mappo_agent import MappoAgent
    from .sim import Simulation
    cfg = copy.deepcopy(BASE)
    cfg.update(copy.deepcopy(scenario))
    cfg['seed'] = seed
    sim = Simulation(cfg)
    buf = []
    for a in sim.agents[:sim.N]:
        assert isinstance(a, MappoAgent)
        a.policy, a.stochastic, a.record = actor, True, buf
    glob, value = {}, {}
    orig = sim._record

    def record(series, t, pos, obs, alive, ever):
        orig(series, t, pos, obs, alive, ever)
        lo, hi = sim.world.bounds
        span = float(np.max(hi - lo))
        g = np.zeros(GLOBAL_DIM)
        for i, a in enumerate(sim.agents[:sim.N]):
            g[4 * i:4 * i + 2] = (a.pos - lo) / span
            g[4 * i + 2] = float(a.alive)
            g[4 * i + 3] = min(len(a.unsecured(t)), 5) / 5.0
        g[-3] = series['coverage'][-1]
        g[-2] = series['tdr_base'][-1]
        g[-1] = t / cfg['duration_s']
        glob[t] = g
        value[t] = (sum(np.exp(-tb / 300.0) for tb in sim.t_base.values()) / max(1, len(sim.targets_xy))
                    + 0.05 * series['coverage'][-1])
    sim._record = record
    summary = sim.run()['summary']
    return buf, glob, value, summary


def build_batch(results, gamma, lam, critic):
    """Per-agent trajectories -> (obs, gstate, act, logp, adv, ret)."""
    O, G, A, LP, ADV, RET = [], [], [], [], [], []
    for buf, glob, value, _ in results:
        ts = sorted(glob)
        by_agent = {}
        for t, i, o, a, lp in buf:
            by_agent.setdefault(i, []).append((t, o, a, lp))
        for i, traj in by_agent.items():
            traj.sort(key=lambda x: x[0])
            obs = np.array([x[1] for x in traj])
            tt = [x[0] for x in traj]
            gs = np.array([glob[min(ts, key=lambda s: abs(s - t))] for t in tt])
            rew = []
            for k, t in enumerate(tt):
                t_next = tt[k + 1] if k + 1 < len(tt) else ts[-1]
                v0 = value[min(ts, key=lambda s: abs(s - t))]
                v1 = value[min(ts, key=lambda s: abs(s - t_next))]
                rew.append((v1 - v0) * 100.0)
            rew = np.array(rew)
            v = critic(np.concatenate([gs, obs], 1))
            v_next = np.append(v[1:], 0.0)
            delta = rew + gamma * v_next - v
            adv = np.zeros_like(rew)
            run = 0.0
            for k in reversed(range(len(rew))):
                run = delta[k] + gamma * lam * run
                adv[k] = run
            O.append(obs); G.append(gs); A.append([x[2] for x in traj])
            LP.append([x[3] for x in traj]); ADV.append(adv); RET.append(adv + v)
    cat = np.concatenate
    return cat(O), cat(G), cat(A), cat(LP), cat(ADV), cat(RET)


def main():
    import torch
    import torch.nn as nn
    from sentinel_core.baselines.mappo_agent import N_ACTIONS, OBS_DIM
    ap = argparse.ArgumentParser()
    ap.add_argument('--iters', type=int, default=120)
    ap.add_argument('--episodes', type=int, default=16)
    ap.add_argument('--jobs', type=int, default=16)
    ap.add_argument('--out', default='configs/mappo_policy.npz')
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--seed0', type=int, default=2000)
    a = ap.parse_args()
    a.out = os.path.abspath(a.out)
    torch.manual_seed(0)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    def mlp(i, o):
        return nn.Sequential(nn.Linear(i, a.hidden), nn.Tanh(), nn.Linear(a.hidden, a.hidden),
                             nn.Tanh(), nn.Linear(a.hidden, o))
    actor = mlp(OBS_DIM, N_ACTIONS).to(dev)
    critic = mlp(GLOBAL_DIM + OBS_DIM, 1).to(dev)
    with torch.no_grad():
        actor[-1].weight.mul_(0.01)
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=3e-4)

    def actor_np():
        ps = []
        for m in actor:
            if isinstance(m, nn.Linear):
                ps += [m.weight.detach().cpu().numpy().T.copy(), m.bias.detach().cpu().numpy().copy()]
        return ps

    def critic_np(x):
        with torch.no_grad():
            return critic(torch.as_tensor(x, dtype=torch.float32, device=dev)).squeeze(-1).cpu().numpy()

    log = []
    seed = a.seed0
    with ProcessPoolExecutor(a.jobs) as pool:
        for it in range(a.iters):
            t0 = time.time()
            params = actor_np()
            jobs = [(params, seed + k, SCENARIOS[k % len(SCENARIOS)]) for k in range(a.episodes)]
            seed += a.episodes
            results = list(pool.map(rollout, jobs))
            O, G, A, LP, ADV, RET = build_batch(results, 0.99, 0.95, critic_np)
            ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
            T = lambda x, dt=torch.float32: torch.as_tensor(x, dtype=dt, device=dev)  # noqa: E731
            O_, G_, A_, LP_, ADV_, RET_ = T(O), T(G), T(A, torch.long), T(LP), T(ADV), T(RET)
            n = len(O)
            for _ in range(4):
                perm = torch.randperm(n, device=dev)
                for s in range(0, n, 4096):
                    idx = perm[s:s + 4096]
                    logits = actor(O_[idx])
                    dist = torch.distributions.Categorical(logits=logits)
                    lp = dist.log_prob(A_[idx])
                    ratio = torch.exp(lp - LP_[idx])
                    pg = -torch.min(ratio * ADV_[idx],
                                    torch.clamp(ratio, 0.8, 1.2) * ADV_[idx]).mean()
                    v = critic(torch.cat([G_[idx], O_[idx]], 1)).squeeze(-1)
                    vf = ((v - RET_[idx]) ** 2).mean()
                    loss = pg + 0.5 * vf - 0.01 * dist.entropy().mean()
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), 0.5)
                    opt.step()
            rv = np.mean([r[3]['report_value'] for r in results])
            tb = np.mean([r[3]['tdr_base'] for r in results])
            log.append((it, rv, tb))
            print(f'iter {it:3d} report_value {rv:.4f} tdr_base {tb:.3f} samples {n} '
                  f'{time.time() - t0:.0f}s', flush=True)
            np.savez(a.out, *actor_np())
            np.save(a.out.replace('.npz', '_curve.npy'), np.array(log))


if __name__ == '__main__':
    main()
