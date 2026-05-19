import argparse
import os

from .config import make_cfg
from .evaluate import evaluate_all
from .plot_results import plot_results
from .train_manager import train_manager
from .train_worker import train_worker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="submission", choices=["small", "submission", "paper"])
    parser.add_argument("--out", default="results_final")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--worker-updates", type=int, default=None)
    parser.add_argument("--manager-updates", type=int, default=None)
    parser.add_argument("--backhaul-budget", type=float, default=None)
    parser.add_argument("--V", type=float, default=None)
    parser.add_argument("--candidate", action="store_true")
    args = parser.parse_args()

    cfg = make_cfg(args.profile)
    if args.worker_updates is not None:
        cfg.worker_updates = args.worker_updates
    if args.manager_updates is not None:
        cfg.manager_updates = args.manager_updates
    if args.backhaul_budget is not None:
        cfg.E_bh_max_per_uav = args.backhaul_budget
        cfg.E_scale = args.backhaul_budget
    if args.V is not None:
        cfg.V = args.V

    os.makedirs(args.out, exist_ok=True)
    cfg.save(os.path.join(args.out, "config.json"))

    train_worker(cfg, args.out)
    train_manager(cfg, os.path.join(args.out, "worker.pt"), args.out)
    evaluate_all(cfg, args.out, episodes=args.episodes, candidate=args.candidate)
    plot_results(args.out, budget=cfg.E_bh_max_per_uav)


if __name__ == "__main__":
    main()
