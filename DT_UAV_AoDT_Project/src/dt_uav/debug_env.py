import argparse
import numpy as np

from .config import make_cfg
from .env import DTUAVAoDTEnv


def nearest_grid_indices(env, points):
    out = []
    for point in points:
        d = np.linalg.norm(env.grid - point, axis=1)
        out.append(int(np.argmin(d)))
    return np.array(out, dtype=np.int64)


def entity_centers(env):
    centers = []
    for e in range(env.E):
        sensors = env.sensor_pos[env.entity_of_sensor == e]
        centers.append(sensors.mean(axis=0))
    return np.array(centers, dtype=np.float32)


def assign_dt_to_nearest_uav(env, pos_idx):
    centers = entity_centers(env)
    q = env.grid[pos_idx]
    hosts = []
    for center in centers:
        hosts.append(int(np.argmin(np.linalg.norm(q - center, axis=1))))
    return np.array(hosts, dtype=np.int64)


def make_good_deployment(env):
    centers = entity_centers(env)
    chosen_entities = np.linspace(0, env.E - 1, env.M).round().astype(int)
    pos_idx = nearest_grid_indices(env, centers[chosen_entities])
    dt_host = assign_dt_to_nearest_uav(env, pos_idx)
    return dt_host, pos_idx


def make_random_deployment(env, seed):
    rng = np.random.default_rng(seed)
    pos_idx = rng.integers(0, env.G, size=env.M, dtype=np.int64)
    dt_host = rng.integers(0, env.M, size=env.E, dtype=np.int64)
    return dt_host, pos_idx


def make_bad_deployment(env):
    centers = entity_centers(env)
    chosen_entities = np.linspace(0, env.E - 1, env.M).round().astype(int)
    pos_idx = []
    for e in chosen_entities:
        d = np.linalg.norm(env.grid - centers[e], axis=1)
        pos_idx.append(int(np.argmax(d)))
    pos_idx = np.array(pos_idx, dtype=np.int64)

    nearest_hosts = assign_dt_to_nearest_uav(env, pos_idx)
    dt_host = (nearest_hosts + 1) % env.M
    return dt_host.astype(np.int64), pos_idx


def run_fixed_deployment(cfg, name, dt_host, pos_idx, seed):
    env = DTUAVAoDTEnv(cfg, seed=seed)
    env.set_manager_action(dt_host, pos_idx)

    slots = cfg.H * cfg.K
    infos = []
    entity_trace = []

    for _ in range(slots):
        action = env.heuristic_worker_action(avoid_energy=False)
        _, _, _, info = env.step_worker(action)
        infos.append(info)
        entity_trace.append(env.entity_aodt().copy())

    entity_trace = np.stack(entity_trace)
    energy = np.mean(np.stack([x["E_bh_m"] for x in infos]), axis=0)
    served = int(sum(x["served"] for x in infos))
    capacity = slots * cfg.M

    return {
        "name": name,
        "dt_host": np.array(dt_host, dtype=np.int64),
        "pos_idx": np.array(pos_idx, dtype=np.int64),
        "positions": env.grid[pos_idx],
        "avg_aodt": float(np.mean([x["avg_aodt"] for x in infos])),
        "p95_aodt": float(np.percentile([x["p95_aodt"] for x in infos], 95)),
        "max_aodt": float(np.max([x["max_aodt"] for x in infos])),
        "served_ratio": float(served / max(capacity, 1)),
        "backhaul_energy_per_uav": energy,
        "successful_dt_updates": served,
        "entity_trace": entity_trace,
    }


def print_result(result):
    print("=" * 80)
    print(f"scenario: {result['name']}")
    print(f"dt_host_by_entity: {result['dt_host'].tolist()}")
    print(f"pos_idx_by_uav: {result['pos_idx'].tolist()}")
    print(f"uav_positions: {np.round(result['positions'], 2).tolist()}")
    print(f"avg_aodt: {result['avg_aodt']:.6f}")
    print(f"p95_aodt: {result['p95_aodt']:.6f}")
    print(f"max_aodt: {result['max_aodt']:.6f}")
    print(f"served_ratio: {result['served_ratio']:.6f}")
    print(
        "backhaul_energy_per_uav: "
        f"{np.round(result['backhaul_energy_per_uav'], 8).tolist()}"
    )
    print(f"successful_dt_updates: {result['successful_dt_updates']}")
    print("per_entity_aodt_over_time:")
    for e in range(result["entity_trace"].shape[1]):
        vals = ", ".join(f"{x:.3f}" for x in result["entity_trace"][:, e])
        print(f"  entity_{e}: [{vals}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="small", choices=["tiny", "small", "paper"])
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = make_cfg(args.profile)
    seed = cfg.seed if args.seed is None else args.seed

    if cfg.M < 2:
        raise ValueError("debug_env requires at least two UAVs; use --profile small or paper")

    reference_env = DTUAVAoDTEnv(cfg, seed=seed)
    scenarios = [
        ("good_deployment", make_good_deployment(reference_env)),
        ("random_deployment", make_random_deployment(reference_env, seed + 999)),
        ("bad_deployment", make_bad_deployment(reference_env)),
    ]

    print(f"profile: {args.profile}")
    print(f"seed: {seed}")
    print(f"slots: {cfg.H * cfg.K}")
    print(f"M={cfg.M}, E={cfg.E}, I={cfg.I}, grid_n={cfg.grid_n}")

    for name, (dt_host, pos_idx) in scenarios:
        result = run_fixed_deployment(cfg, name, dt_host, pos_idx, seed)
        print_result(result)


if __name__ == "__main__":
    main()
