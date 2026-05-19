import argparse
import os
import pandas as pd
import torch
from .config import make_cfg
from .env import DTUAVAoDTEnv
from .ppo import ActorCritic, PPOTrainer, RolloutBuffer, save_checkpoint


def train_worker(cfg, out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    env = DTUAVAoDTEnv(cfg, seed=cfg.seed)
    model = ActorCritic(len(env.worker_state()), env.worker_action_dim)
    trainer = PPOTrainer(
        model,
        lr=cfg.lr_worker,
        clip_eps=cfg.clip_eps,
        ppo_epochs=cfg.ppo_epochs,
        batch_size=cfg.batch_size,
        vf_coef=cfg.vf_coef,
        ent_coef=cfg.ent_coef_worker,
        max_grad_norm=cfg.max_grad_norm,
    )
    state = env.reset()
    rows = []
    for upd in range(cfg.worker_updates):
        buf = RolloutBuffer()
        infos = []
        for _ in range(cfg.worker_steps_per_update):
            # Randomize slow deployment occasionally so the worker generalizes.
            if env.t % cfg.H == 0:
                a_m = env.rng.integers(0, env.manager_action_dim)
                dt_host, pos_idx = env.decode_manager_action(a_m)
                env.set_manager_action(dt_host, pos_idx)
            action_mask = env.valid_worker_action_mask()
            action, logp, value = model.act(state, action_mask=action_mask)
            next_state, reward, done, info = env.step_worker(action)
            buf.add(state, action, logp, reward, done, value, action_mask=action_mask)
            state = next_state
            infos.append(info)
        with torch.no_grad():
            _, last_value = model(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(next(model.parameters()).device))
        returns, adv = buf.compute_returns_advantages(float(last_value.item()), cfg.gamma, cfg.lam)
        loss_info = trainer.update(buf, returns, adv)
        row = {
            "update": upd,
            "avg_aodt": sum(x["avg_aodt"] for x in infos) / len(infos),
            "p95_aodt": sum(x["p95_aodt"] for x in infos) / len(infos),
            "max_aodt": max(x["max_aodt"] for x in infos),
            "served_per_slot": sum(x["served"] for x in infos) / len(infos),
            "invalid_per_slot": sum(x["invalid"] for x in infos) / len(infos),
            "loss": loss_info["loss"],
        }
        rows.append(row)
        print(f"worker {upd+1}/{cfg.worker_updates} avg_aodt={row['avg_aodt']:.3f} served={row['served_per_slot']:.2f} invalid={row['invalid_per_slot']:.2f}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "worker_train_log.csv"), index=False)
    save_checkpoint(os.path.join(out_dir, "worker.pt"), model, cfg, {"train_log": rows})
    return model, df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default="small", choices=["tiny", "small", "paper"])
    p.add_argument("--out", default="results")
    args = p.parse_args()
    cfg = make_cfg(args.profile)
    train_worker(cfg, args.out)


if __name__ == "__main__":
    main()
