from .config import make_cfg
from .train_worker import train_worker
from .train_manager import train_manager
from .evaluate import evaluate_all


def main():
    cfg = make_cfg("tiny")
    cfg.worker_updates = 2
    cfg.manager_updates = 2
    cfg.worker_steps_per_update = 128
    cfg.manager_episodes_per_update = 2
    out = "results_quickstart"
    train_worker(cfg, out)
    train_manager(cfg, f"{out}/worker.pt", out)
    evaluate_all(cfg, out, episodes=2, candidate=False)


if __name__ == "__main__":
    main()
