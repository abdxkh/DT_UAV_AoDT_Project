import itertools
import numpy as np
from .env import DTUAVAoDTEnv, ManagerEnv


def nearest_grid_indices(env, points):
    out = []
    for p in points:
        d = np.linalg.norm(env.grid - p, axis=1)
        out.append(int(np.argmin(d)))
    return np.array(out, dtype=np.int64)


def kmeans_positions(env, iters=20):
    pts = env.sensor_pos
    M = env.M
    centers = pts[np.linspace(0, len(pts)-1, M).round().astype(int)].copy()
    for _ in range(iters):
        labels = np.argmin(((pts[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2), axis=1)
        for m in range(M):
            if np.any(labels == m):
                centers[m] = pts[labels == m].mean(axis=0)
    return nearest_grid_indices(env, centers)


def manager_action_for_baseline(env, name):
    name = name.lower()
    if name == "random":
        return int(env.rng.integers(0, env.manager_action_dim))
    if name == "kmeans":
        pos_idx = kmeans_positions(env)
        # host each entity on nearest UAV position
        q = env.grid[pos_idx]
        dt_host = []
        for e in range(env.E):
            center = env.sensor_pos[env.entity_of_sensor == e].mean(axis=0)
            dt_host.append(int(np.argmin(np.linalg.norm(q - center, axis=1))))
        return env.encode_manager_action(np.array(dt_host), pos_idx)
    if name == "aodt_greedy":
        entity_order = np.argsort(-env.entity_aodt())
        pos_idx = []
        for m in range(env.M):
            e = entity_order[m % env.E]
            center = env.sensor_pos[env.entity_of_sensor == e].mean(axis=0)
            pos_idx.append(nearest_grid_indices(env, [center])[0])
        pos_idx = np.array(pos_idx, dtype=np.int64)
        q = env.grid[pos_idx]
        dt_host = []
        for e in range(env.E):
            center = env.sensor_pos[env.entity_of_sensor == e].mean(axis=0)
            dt_host.append(int(np.argmin(np.linalg.norm(q - center, axis=1))))
        return env.encode_manager_action(np.array(dt_host), pos_idx)
    raise ValueError("Unknown baseline " + name)


def evaluate_fixed_policy(cfg, manager_policy=None, worker_policy=None, baseline=None, episodes=5, seed=123):
    rows = []
    for ep in range(episodes):
        env = ManagerEnv(cfg, worker_policy=worker_policy, seed=seed + ep)
        state = env.reset()
        done = False
        infos = []
        while not done:
            if baseline:
                action = manager_action_for_baseline(env.fast, baseline)
            else:
                action = manager_policy.act_deterministic(state)
            state, reward, done, info = env.step(action)
            infos.append(info)
        E = np.mean(np.stack([x["E_bh_m"] for x in infos]), axis=0)
        rows.append({
            "policy": baseline or "lya_hppo",
            "episode": ep,
            "avg_aodt": float(np.mean([x["avg_aodt"] for x in infos])),
            "p95_aodt": float(np.percentile([x["p95_aodt"] for x in infos], 95)),
            "max_aodt": float(np.max([x["max_aodt"] for x in infos])),
            "mean_backhaul_energy": float(np.mean(E)),
            "max_backhaul_energy": float(np.max(E)),
            "feasible": bool(np.all(E <= cfg.E_bh_max_per_uav + 1e-9)),
            "max_Z": float(np.max([x["max_Z"] for x in infos])),
        })
    return rows


def candidate_site_search(cfg, worker_policy=None, seed=777, max_actions=None):
    """Small-instance optimized baseline: enumerate manager actions on the grid.

    This is meant for tiny/small settings only. It is expensive for paper-scale.
    """
    env = ManagerEnv(cfg, worker_policy=worker_policy, seed=seed)
    action_dim = env.action_dim
    if max_actions is None or max_actions >= action_dim:
        actions = range(action_dim)
    else:
        rng = np.random.default_rng(seed)
        actions = rng.choice(action_dim, size=max_actions, replace=False)

    best = None
    for a in actions:
        env = ManagerEnv(cfg, worker_policy=worker_policy, seed=seed)
        s = env.reset()
        infos = []
        done = False
        # Use same candidate action for all windows for a stable deployment baseline.
        while not done:
            s, r, done, info = env.step(int(a))
            infos.append(info)
        avg_aodt = float(np.mean([x["avg_aodt"] for x in infos]))
        E = np.mean(np.stack([x["E_bh_m"] for x in infos]), axis=0)
        feasible = bool(np.all(E <= cfg.E_bh_max_per_uav + 1e-9))
        score = avg_aodt + (0 if feasible else 1e6)
        rec = {"action": int(a), "avg_aodt": avg_aodt, "E_bh_m": E, "feasible": feasible, "score": score}
        if best is None or rec["score"] < best["score"]:
            best = rec
    return best
