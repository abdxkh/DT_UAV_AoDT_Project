import itertools
import numpy as np


class DTUAVAoDTEnv:
    """Fast-timescale simulator for a DT-assisted UAV edge network.

    Naming note:
    - aodt_i is the Age of Digital Twin for sensor view i at the DT host.
    - aodt_e is the entity-level AoDT, max over all sensors monitoring entity e.
    """

    def __init__(self, cfg, seed=None):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed if seed is None else seed)
        self.M, self.E, self.I = cfg.M, cfg.E, cfg.I
        self.grid = self._make_grid()
        self.G = len(self.grid)
        self.worker_action_dim = ((self.I + 1) * len(cfg.p_levels)) ** self.M
        self.manager_action_dim = (self.G ** self.M) * (self.M ** self.E)
        self.entity_of_sensor = np.arange(self.I) % self.E
        self._build_positions()
        self.reset()

    def _make_grid(self):
        xs = np.linspace(0.15, 0.85, self.cfg.grid_n) * self.cfg.area_size
        return np.array([[x, y] for x in xs for y in xs], dtype=np.float32)

    def _build_positions(self):
        centers = []
        angles = np.linspace(0, 2 * np.pi, self.E, endpoint=False)
        radius = self.cfg.area_size * 0.27
        c = self.cfg.area_size / 2
        for a in angles:
            centers.append([c + radius * np.cos(a), c + radius * np.sin(a)])
        self.entity_pos = np.array(centers, dtype=np.float32)
        sensors = []
        for i in range(self.I):
            e = self.entity_of_sensor[i]
            jitter = self.rng.normal(0.0, self.cfg.area_size * 0.035, size=2)
            sensors.append(np.clip(self.entity_pos[e] + jitter, 0, self.cfg.area_size))
        self.sensor_pos = np.array(sensors, dtype=np.float32)

    def reset(self):
        self.t = 0
        self.Q = np.zeros(self.I, dtype=np.float32)
        self.W = np.zeros(self.I, dtype=np.float32)
        self.U = np.zeros(self.I, dtype=np.float32)
        self.aodt_i = np.ones(self.I, dtype=np.float32)
        self.prev_served = np.zeros(self.I, dtype=np.float32)
        self.Z = np.zeros(self.M, dtype=np.float32)
        self.dt_host = np.arange(self.E, dtype=np.int64) % self.M
        self.pos_idx = np.linspace(0, self.G - 1, self.M).round().astype(np.int64)
        self.q = self.grid[self.pos_idx].copy()
        return self.worker_state()

    def set_manager_action(self, dt_host, pos_idx):
        self.dt_host = np.asarray(dt_host, dtype=np.int64) % self.M
        self.pos_idx = np.asarray(pos_idx, dtype=np.int64) % self.G
        self.q = self.grid[self.pos_idx].copy()

    def decode_worker_action(self, action):
        base = (self.I + 1) * len(self.cfg.p_levels)
        x = int(action)
        choices = []
        for _ in range(self.M):
            d = x % base
            x //= base
            sensor_choice = d // len(self.cfg.p_levels)   # 0 means idle, 1..I are sensors
            p_idx = d % len(self.cfg.p_levels)
            sensor = int(sensor_choice) - 1
            choices.append((sensor, int(p_idx)))
        return choices

    def decode_manager_action(self, action):
        x = int(action)
        pos_idx = []
        for _ in range(self.M):
            pos_idx.append(x % self.G)
            x //= self.G
        dt_host = []
        for _ in range(self.E):
            dt_host.append(x % self.M)
            x //= self.M
        return np.array(dt_host, dtype=np.int64), np.array(pos_idx, dtype=np.int64)

    def encode_manager_action(self, dt_host, pos_idx):
        x = 0
        mult = 1
        for m in range(self.M):
            x += int(pos_idx[m]) * mult
            mult *= self.G
        for e in range(self.E):
            x += int(dt_host[e]) * mult
            mult *= self.M
        return int(x)

    def entity_aodt(self):
        vals = []
        for e in range(self.E):
            vals.append(float(np.max(self.aodt_i[self.entity_of_sensor == e])))
        return np.array(vals, dtype=np.float32)

    def avg_aodt(self):
        return float(np.mean(self.entity_aodt()))

    def _channel_gain(self, a, b):
        horizontal = float(np.linalg.norm(a - b))
        d = np.sqrt(horizontal * horizontal + self.cfg.uav_altitude ** 2)
        return self.cfg.beta0 / (d ** self.cfg.pathloss_alpha + 1e-12)

    def _rate(self, B, p, h):
        snr = p * h / max(self.cfg.N0 * B, 1e-30)
        return B * np.log2(1.0 + snr)

    def _delay_and_backhaul_energy(self, i, m, p_ul):
        e = self.entity_of_sensor[i]
        host = int(self.dt_host[e])
        bits = max(float(self.W[i]), 0.0)
        h_ul = self._channel_gain(self.sensor_pos[i], self.q[m])
        r_ul = max(self._rate(self.cfg.B_ac, p_ul, h_ul), 1e-12)
        tau_ul = bits / r_ul

        tau_bh = 0.0
        E_bh = 0.0
        if host != m:
            h_bh = self._channel_gain(self.q[m], self.q[host])
            r_bh = max(self._rate(self.cfg.B_bh, self.cfg.p_bh, h_bh), 1e-12)
            tau_bh = bits / r_bh
            E_bh = self.cfg.p_bh * tau_bh

        tau_proc = self.cfg.kappa_cycles_per_bit * bits / self.cfg.f_cpu
        delay_slots = (tau_ul + tau_bh + tau_proc) / self.cfg.T_s
        return delay_slots, E_bh

    def _start_slot_arrivals(self):
        old_Q = self.Q.copy()
        old_U = self.U.copy()
        A = (self.rng.random(self.I) < self.cfg.arrival_prob).astype(np.float32)
        sizes = self.rng.exponential(self.cfg.packet_bits_mean, size=self.I).astype(np.float32)
        D = A * sizes

        self.Q = np.maximum(self.Q * (1.0 - self.prev_served), A)
        self.W = D + (1.0 - A) * self.W * (1.0 - self.prev_served)
        self.U = np.where(
            A == 1.0,
            0.0,
            np.where(old_Q == 1.0, (old_U + 1.0) * (1.0 - self.prev_served), 0.0),
        ).astype(np.float32)
        return A, D

    def worker_state(self):
        parts = [
            self.Q,
            np.log1p(self.W / max(self.cfg.packet_bits_mean, 1.0)),
            self.U / 10.0,
            self.aodt_i / 10.0,
            self.Z / max(self.cfg.Z_clip, 1.0),
            self.dt_host[self.entity_of_sensor].astype(np.float32) / max(1, self.M - 1),
            self.sensor_pos.flatten() / self.cfg.area_size,
            self.q.flatten() / self.cfg.area_size,
        ]
        return np.concatenate(parts).astype(np.float32)

    def manager_state(self):
        x = np.zeros((self.E, self.M), dtype=np.float32)
        for e in range(self.E):
            x[e, self.dt_host[e]] = 1.0
        parts = [
            self.entity_aodt() / 10.0,
            self.aodt_i / 10.0,
            self.Z / max(self.cfg.Z_clip, 1.0),
            self.q.flatten() / self.cfg.area_size,
            x.flatten(),
            self.sensor_pos.flatten() / self.cfg.area_size,
        ]
        return np.concatenate(parts).astype(np.float32)

    def step_worker(self, action):
        A, D = self._start_slot_arrivals()
        choices = self.decode_worker_action(action)
        served = np.zeros(self.I, dtype=np.float32)
        E_bh_m = np.zeros(self.M, dtype=np.float32)
        invalid = 0
        selected = set()
        delay_by_sensor = np.zeros(self.I, dtype=np.float32)

        for m, (sensor, p_idx) in enumerate(choices):
            if sensor < 0:
                continue
            if sensor >= self.I or self.Q[sensor] <= 0 or sensor in selected:
                invalid += 1
                continue
            selected.add(sensor)
            p_ul = float(self.cfg.p_levels[p_idx])
            delay_slots, e_bh = self._delay_and_backhaul_energy(sensor, m, p_ul)
            served[sensor] = 1.0
            delay_by_sensor[sensor] = delay_slots
            E_bh_m[m] += e_bh

        # AoDT update: if the DT receives the update, age resets to waiting time + end-to-end delay.
        self.aodt_i = (1.0 - served) * (self.aodt_i + 1.0) + served * (self.U + delay_by_sensor)
        self.prev_served = served
        self.t += 1

        avg_aodt = self.avg_aodt()
        pending = float(np.sum(self.Q))
        idle_penalty = self.cfg.idle_pending_penalty * max(0.0, pending - float(np.sum(served)))
        reward = -avg_aodt / self.cfg.aodt_scale - self.cfg.invalid_penalty * invalid - idle_penalty

        info = {
            "avg_aodt": avg_aodt,
            "p95_aodt": float(np.percentile(self.entity_aodt(), 95)),
            "max_aodt": float(np.max(self.entity_aodt())),
            "sensor_avg_aodt": float(np.mean(self.aodt_i)),
            "served": int(np.sum(served)),
            "invalid": int(invalid),
            "E_bh_m": E_bh_m.copy(),
            "E_bh_sum": float(np.sum(E_bh_m)),
            "A": A,
            "D": D,
            "served_vec": served.copy(),
        }
        return self.worker_state(), float(reward), False, info

    def heuristic_worker_action(self, avoid_energy=False):
        """Simple AoDT-aware scheduling. Used for debugging and baselines."""
        choices = []
        used = set()
        for m in range(self.M):
            best_i, best_score = -1, -1e18
            for i in range(self.I):
                if i in used or self.Q[i] <= 0:
                    continue
                p_idx = len(self.cfg.p_levels) - 1
                delay, e_bh = self._delay_and_backhaul_energy(i, m, float(self.cfg.p_levels[p_idx]))
                score = self.aodt_i[i] + 0.4 * self.U[i] - 0.1 * delay
                if avoid_energy:
                    score -= 2.0 * max(0.0, (self.Z[m] / max(self.cfg.Z_clip, 1.0))) * e_bh / max(self.cfg.E_scale, 1e-12)
                if score > best_score:
                    best_i, best_score = i, score
            if best_i >= 0:
                used.add(best_i)
                choices.append((best_i, len(self.cfg.p_levels) - 1))
            else:
                choices.append((-1, 0))
        return self.encode_worker_choices(choices)

    def encode_worker_choices(self, choices):
        base = (self.I + 1) * len(self.cfg.p_levels)
        x = 0
        mult = 1
        for sensor, p_idx in choices:
            sensor_choice = int(sensor) + 1 if sensor >= 0 else 0
            d = sensor_choice * len(self.cfg.p_levels) + int(p_idx)
            x += d * mult
            mult *= base
        return int(x)


class ManagerEnv:
    def __init__(self, cfg, worker_policy=None, seed=None):
        self.cfg = cfg
        self.fast = DTUAVAoDTEnv(cfg, seed=seed)
        self.worker_policy = worker_policy
        self.k = 0

    @property
    def action_dim(self):
        return self.fast.manager_action_dim

    @property
    def state_dim(self):
        return len(self.fast.manager_state())

    def reset(self):
        self.fast.reset()
        self.k = 0
        return self.fast.manager_state()

    def step(self, action):
        dt_host, pos_idx = self.fast.decode_manager_action(action)
        self.fast.set_manager_action(dt_host, pos_idx)
        old_Z = self.fast.Z.copy()
        infos = []
        for _ in range(self.cfg.H):
            if self.worker_policy is None:
                a = self.fast.heuristic_worker_action(avoid_energy=False)
            else:
                s = self.fast.worker_state()
                a = self.worker_policy.act_deterministic(s)
            _, _, _, info = self.fast.step_worker(a)
            infos.append(info)

        avg_aodt = float(np.mean([x["avg_aodt"] for x in infos]))
        p95_aodt = float(np.percentile([x["p95_aodt"] for x in infos], 95))
        max_aodt = float(np.max([x["max_aodt"] for x in infos]))
        E_window = np.mean(np.stack([x["E_bh_m"] for x in infos]), axis=0)
        excess = E_window - self.cfg.E_bh_max_per_uav
        self.fast.Z = np.maximum(self.fast.Z + excess / max(self.cfg.E_scale, 1e-12), 0.0).astype(np.float32)

        lyap_cost = float(np.dot(old_Z, excess / max(self.cfg.E_scale, 1e-12)))
        reward = -self.cfg.V * avg_aodt / self.cfg.aodt_scale - lyap_cost
        self.k += 1
        done = self.k >= self.cfg.K
        info = {
            "avg_aodt": avg_aodt,
            "p95_aodt": p95_aodt,
            "max_aodt": max_aodt,
            "E_bh_m": E_window,
            "max_Z": float(np.max(self.fast.Z)),
            "mean_Z": float(np.mean(self.fast.Z)),
            "dt_host": dt_host.copy(),
            "pos_idx": pos_idx.copy(),
        }
        return self.fast.manager_state(), float(reward), done, info
