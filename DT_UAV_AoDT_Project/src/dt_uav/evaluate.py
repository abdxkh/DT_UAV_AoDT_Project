import argparse
import os
import pandas as pd
import torch
from .config import make_cfg
from .env import DTUAVAoDTEnv
from .ppo import ActorCritic, load_checkpoint
from .baselines import evaluate_fixed_policy, candidate_site_search


def load_models(cfg, out_dir):
    dummy = DTUAVAoDTEnv(cfg)
    worker = ActorCritic(len(dummy.worker_state()), dummy.worker_action_dim)
    manager = ActorCritic(len(dummy.manager_state()), dummy.manager_action_dim)
    worker_path = os.path.join(out_dir, "worker.pt")
    manager_path = os.path.join(out_dir, "manager.pt")
    if os.path.exists(worker_path):
        load_checkpoint(worker_path, worker)
    else:
        worker = None
    if os.path.exists(manager_path):
        load_checkpoint(manager_path, manager)
    else:
        manager = None
    return worker, manager


def evaluate_all(cfg, out_dir="results", episodes=5, candidate=False):
    worker, manager = load_models(cfg, out_dir)
    rows = []
    if manager is not None:
        rows += evaluate_fixed_policy(cfg, manager_policy=manager, worker_policy=worker, episodes=episodes)
    for b in ["kmeans", "aodt_greedy", "random"]:
        rows += evaluate_fixed_policy(cfg, worker_policy=worker, baseline=b, episodes=episodes)
    df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "evaluation.csv"), index=False)
    print(df.groupby("policy")[["avg_aodt", "p95_aodt", "max_aodt", "mean_backhaul_energy", "feasible"]].mean())
    if candidate:
        best = candidate_site_search(cfg, worker_policy=worker, max_actions=2000)
        print("Best candidate-site deployment:", best)
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default="small", choices=["tiny", "small", "paper"])
    p.add_argument("--out", default="results")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--candidate", action="store_true")
    args = p.parse_args()
    cfg = make_cfg(args.profile)
    evaluate_all(cfg, args.out, args.episodes, args.candidate)


if __name__ == "__main__":
    main()
