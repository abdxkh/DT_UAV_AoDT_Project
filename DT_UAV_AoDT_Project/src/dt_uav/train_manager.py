import argparse
import os
import pandas as pd
import torch
from .config import make_cfg
from .env import DTUAVAoDTEnv, ManagerEnv
from .ppo import ActorCritic, PPOTrainer, RolloutBuffer, load_checkpoint, save_checkpoint


def train_manager(cfg, worker_ckpt=None, out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(cfg.seed + 1000)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed + 1000)
    dummy = DTUAVAoDTEnv(cfg, seed=cfg.seed)
    worker = ActorCritic(len(dummy.worker_state()), dummy.worker_action_dim)
    if worker_ckpt and os.path.exists(worker_ckpt):
        load_checkpoint(worker_ckpt, worker)
        print("Loaded worker:", worker_ckpt)
    else:
        worker = None
        print("No worker checkpoint found. Using heuristic worker inside manager env.")

    env = ManagerEnv(cfg, worker_policy=worker, seed=cfg.seed + 100)
    model = ActorCritic(env.state_dim, env.action_dim)
    trainer = PPOTrainer(
        model,
        lr=cfg.lr_manager,
        clip_eps=cfg.clip_eps,
        ppo_epochs=cfg.ppo_epochs,
        batch_size=cfg.batch_size,
        vf_coef=cfg.vf_coef,
        ent_coef=cfg.ent_coef_manager,
        max_grad_norm=cfg.max_grad_norm,
    )
    rows = []
    best_score = None
    best_payload = None
    for upd in range(cfg.manager_updates):
        buf = RolloutBuffer()
        infos = []
        for _ in range(cfg.manager_episodes_per_update):
            state = env.reset()
            done = False
            while not done:
                action, logp, value = model.act(state)
                next_state, reward, done, info = env.step(action)
                buf.add(state, action, logp, reward, done, value)
                state = next_state
                infos.append(info)
        returns, adv = buf.compute_returns_advantages(0.0, cfg.gamma, cfg.lam)
        loss_info = trainer.update(buf, returns, adv)
        row = {
            "update": upd,
            "avg_aodt": sum(x["avg_aodt"] for x in infos) / len(infos),
            "p95_aodt": sum(x["p95_aodt"] for x in infos) / len(infos),
            "max_aodt": max(x["max_aodt"] for x in infos),
            "mean_backhaul_energy": sum(float(x["E_bh_m"].mean()) for x in infos) / len(infos),
            "max_backhaul_energy": max(float(x["E_bh_m"].max()) for x in infos),
            "soft_violation": sum(float(x["soft_violation"]) for x in infos) / len(infos),
            "max_Z": max(x["max_Z"] for x in infos),
            "loss": loss_info["loss"],
        }
        violation = max(0.0, row["max_backhaul_energy"] - cfg.E_bh_max_per_uav)
        score = row["avg_aodt"] + 10000.0 * violation
        if best_score is None or score < best_score:
            best_score = score
            best_payload = {"train_log": rows + [row], "selected_update": upd, "selection_score": score}
            save_checkpoint(os.path.join(out_dir, "manager_best.pt"), model, cfg, best_payload)
        rows.append(row)
        print(f"manager {upd+1}/{cfg.manager_updates} avg_aodt={row['avg_aodt']:.3f} E_bh={row['mean_backhaul_energy']:.5f} maxZ={row['max_Z']:.2f}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "manager_train_log.csv"), index=False)
    save_checkpoint(os.path.join(out_dir, "manager.pt"), model, cfg, {"train_log": rows})
    return model, df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default="small", choices=["tiny", "small", "submission", "paper"])
    p.add_argument("--out", default="results")
    p.add_argument("--worker", default="results/worker.pt")
    args = p.parse_args()
    cfg = make_cfg(args.profile)
    train_manager(cfg, args.worker, args.out)


if __name__ == "__main__":
    main()
