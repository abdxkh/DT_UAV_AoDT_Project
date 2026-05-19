from dataclasses import dataclass, asdict
import json


@dataclass
class Config:
    # Network size
    M: int = 2              # UAVs
    E: int = 3              # physical entities / digital twins
    I: int = 6              # sensors

    # Time
    T_s: float = 0.5        # slot duration in seconds
    H: int = 10             # fast slots per manager window
    K: int = 15             # manager windows per episode

    # Traffic
    arrival_prob: float = 1.0
    packet_bits_mean: float = 80_000.0

    # Geometry
    area_size: float = 600.0
    grid_n: int = 3
    uav_altitude: float = 80.0

    # Communication
    B_ac: float = 1.0e6
    B_bh: float = 1.0e6
    N0: float = 1.0e-17
    beta0: float = 1.0e-3
    pathloss_alpha: float = 2.2
    p_levels: tuple = (0.03, 0.07, 0.12)
    p_bh: float = 0.08

    # Processing
    kappa_cycles_per_bit: float = 300.0
    f_cpu: float = 2.0e9

    # Lyapunov backhaul energy constraint, per UAV per slot
    E_bh_max_per_uav: float = 0.015
    E_scale: float = 0.015
    Z_clip: float = 50.0
    V: float = 10.0

    # Reward scaling
    aodt_scale: float = 10.0
    invalid_penalty: float = 0.8
    idle_pending_penalty: float = 0.03
    manager_bh_soft_weight: float = 0.0

    # PPO
    gamma: float = 0.98
    lam: float = 0.95
    clip_eps: float = 0.2
    ppo_epochs: int = 4
    batch_size: int = 128
    vf_coef: float = 0.5
    ent_coef_worker: float = 0.015
    ent_coef_manager: float = 0.010
    lr_worker: float = 2e-4
    lr_manager: float = 2e-4
    max_grad_norm: float = 0.7

    # Training lengths - intentionally light defaults for Colab smoke tests
    worker_updates: int = 20
    worker_steps_per_update: int = 512
    manager_updates: int = 20
    manager_episodes_per_update: int = 4

    seed: int = 7

    def to_dict(self):
        d = asdict(self)
        d["p_levels"] = list(self.p_levels)
        return d

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


def make_cfg(profile="small"):
    cfg = Config()
    if profile == "tiny":
        cfg.M, cfg.E, cfg.I = 1, 1, 2
        cfg.H, cfg.K = 8, 8
        cfg.area_size = 300.0
        cfg.E_bh_max_per_uav = 0.02
        cfg.E_scale = 0.02
        cfg.worker_updates = 5
        cfg.manager_updates = 5
    elif profile == "small":
        cfg.M, cfg.E, cfg.I = 2, 3, 6
        cfg.H, cfg.K = 10, 15
        cfg.area_size = 600.0
        cfg.E_bh_max_per_uav = 0.015
        cfg.E_scale = 0.015
    elif profile == "paper":
        cfg.M, cfg.E, cfg.I = 3, 5, 15
        cfg.H, cfg.K = 20, 25
        cfg.area_size = 800.0
        cfg.E_bh_max_per_uav = 0.05
        cfg.E_scale = 0.05
        cfg.worker_updates = 80
        cfg.manager_updates = 80
        cfg.worker_steps_per_update = 1024
        cfg.manager_episodes_per_update = 6
    else:
        raise ValueError(f"Unknown profile: {profile}")
    return cfg
