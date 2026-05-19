import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


def mlp(in_dim, out_dim, hidden=128):
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.Tanh(),
        nn.Linear(hidden, hidden), nn.Tanh(),
        nn.Linear(hidden, out_dim),
    )


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.actor = mlp(state_dim, action_dim, hidden)
        self.critic = mlp(state_dim, 1, hidden)

    def forward(self, x):
        logits = self.actor(x)
        value = self.critic(x).squeeze(-1)
        return logits, value

    @torch.no_grad()
    def act_deterministic(self, state, action_mask=None):
        device = next(self.parameters()).device
        s = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        logits, _ = self.forward(s)
        if action_mask is not None:
            mask = torch.as_tensor(action_mask, dtype=torch.bool, device=device).unsqueeze(0)
            logits = logits.masked_fill(~mask, -1e9)
        return int(torch.argmax(logits, dim=-1).item())

    @torch.no_grad()
    def act(self, state, action_mask=None):
        device = next(self.parameters()).device
        s = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        logits, value = self.forward(s)
        if action_mask is not None:
            mask = torch.as_tensor(action_mask, dtype=torch.bool, device=device).unsqueeze(0)
            logits = logits.masked_fill(~mask, -1e9)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return int(action.item()), float(dist.log_prob(action).item()), float(value.item())


class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logps = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.action_masks = []

    def add(self, state, action, logp, reward, done, value, action_mask=None):
        self.states.append(np.array(state, dtype=np.float32))
        self.actions.append(int(action))
        self.logps.append(float(logp))
        self.rewards.append(float(reward))
        self.dones.append(float(done))
        self.values.append(float(value))
        if action_mask is None:
            self.action_masks.append(None)
        else:
            self.action_masks.append(np.array(action_mask, dtype=np.bool_))

    def clear(self):
        self.__init__()

    def compute_returns_advantages(self, last_value, gamma, lam):
        rewards = np.array(self.rewards, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)
        values = np.array(self.values + [last_value], dtype=np.float32)
        adv = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * values[t + 1] * nonterminal - values[t]
            gae = delta + gamma * lam * nonterminal * gae
            adv[t] = gae
        returns = adv + values[:-1]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return returns.astype(np.float32), adv.astype(np.float32)


class PPOTrainer:
    def __init__(self, model, lr=2e-4, clip_eps=0.2, ppo_epochs=4, batch_size=128,
                 vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.7, device=None):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.opt = optim.Adam(self.model.parameters(), lr=lr)
        self.clip_eps = clip_eps
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm

    def update(self, buf, returns, advantages):
        states = torch.as_tensor(np.array(buf.states), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(buf.actions, dtype=torch.long, device=self.device)
        old_logps = torch.as_tensor(buf.logps, dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        if buf.action_masks and buf.action_masks[0] is not None:
            action_masks = torch.as_tensor(np.array(buf.action_masks), dtype=torch.bool, device=self.device)
        else:
            action_masks = None
        n = states.shape[0]
        idxs = np.arange(n)
        losses = []
        for _ in range(self.ppo_epochs):
            np.random.shuffle(idxs)
            for start in range(0, n, self.batch_size):
                mb = idxs[start:start+self.batch_size]
                logits, values = self.model(states[mb])
                if action_masks is not None:
                    logits = logits.masked_fill(~action_masks[mb], -1e9)
                dist = Categorical(logits=logits)
                logps = dist.log_prob(actions[mb])
                entropy = dist.entropy().mean()
                ratio = torch.exp(logps - old_logps[mb])
                surr1 = ratio * advantages[mb]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages[mb]
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = ((returns[mb] - values) ** 2).mean()
                loss = actor_loss + self.vf_coef * critic_loss - self.ent_coef * entropy
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.opt.step()
                losses.append(float(loss.item()))
        return {"loss": float(np.mean(losses)) if losses else 0.0}


def save_checkpoint(path, model, cfg, extra=None):
    payload = {
        "model_state_dict": model.state_dict(),
        "cfg": cfg.to_dict() if hasattr(cfg, "to_dict") else cfg.__dict__,
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_checkpoint(path, model, map_location=None):
    ckpt = torch.load(path, map_location=map_location or "cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    return ckpt
