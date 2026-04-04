"""
Tabular Q-learning policy for alert escalation (demo MDP).
States: risk bucket (0-9), cooldown bucket (0-3).
Actions: 0=none, 1=notify_operator, 2=broadcast_public
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class AlertEscalationEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.MultiDiscrete([10, 4])
        self.action_space = spaces.Discrete(3)
        self._state = (0, 0)
        self._last_action = 0
        self._episode_len = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._state = (
            int(self.np_random.integers(0, 10)),
            int(self.np_random.integers(0, 4)),
        )
        self._last_action = 0
        self._episode_len = 0
        return np.array(self._state, dtype=np.int32), {}

    def step(self, action: int):
        risk_b, cool_b = self._state
        risk = risk_b / 9.0
        reward = 0.0

        # Prefer alerting when risk high; penalize spam and public broadcast when low
        if action == 0:
            if risk > 0.72:
                reward -= 0.8
            else:
                reward += 0.15
        elif action == 1:
            if risk > 0.55:
                reward += 1.2
            else:
                reward -= 0.6
            if self._last_action == 1 and cool_b < 2:
                reward -= 0.4
        else:  # broadcast
            if risk > 0.82:
                reward += 1.0
            else:
                reward -= 1.4

        self._last_action = action
        new_risk = int(np.clip(risk_b + self.np_random.integers(-1, 2), 0, 9))
        new_cool = int(np.clip(cool_b + (1 if action != 0 else -1), 0, 3))
        self._state = (new_risk, new_cool)
        self._episode_len += 1
        terminated = self._episode_len >= 20
        return (
            np.array(self._state, dtype=np.int32),
            float(reward),
            terminated,
            False,
            {},
        )


def train_q_table(
    episodes: int = 2500,
    alpha: float = 0.25,
    gamma: float = 0.92,
    eps: float = 0.25,
) -> np.ndarray:
    env = AlertEscalationEnv()
    Q = np.zeros((10, 4, 3), dtype=np.float32)
    for _ in range(episodes):
        s, _ = env.reset()
        done = False
        while not done:
            rb, cb = int(s[0]), int(s[1])
            if env.np_random.random() < eps:
                a = int(env.action_space.sample())
            else:
                a = int(np.argmax(Q[rb, cb]))
            s2, r, done, _, _ = env.step(a)
            rb2, cb2 = int(s2[0]), int(s2[1])
            td = r + gamma * float(np.max(Q[rb2, cb2])) - Q[rb, cb, a]
            Q[rb, cb, a] += alpha * td
            s = s2
    return Q


def load_or_train_q(path: Path) -> np.ndarray:
    if path.exists():
        return np.load(path)
    q = train_q_table()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, q)
    return q


def rl_action(
    risk_score_0_100: float,
    cooldown_steps: int,
    Q: np.ndarray,
) -> Tuple[int, str]:
    """Map continuous risk + app cooldown into discrete policy action."""
    rb = int(np.clip(risk_score_0_100 / 100.0 * 9.999, 0, 9))
    cb = int(np.clip(cooldown_steps, 0, 3))
    a = int(np.argmax(Q[rb, cb]))
    names = {0: "none", 1: "notify_operator", 2: "broadcast_public"}
    return a, names[a]


def save_policy_json(Q: np.ndarray, path: Path) -> None:
    summary = {
        "shape": list(Q.shape),
        "mean_value": float(Q.mean()),
        "max_value": float(Q.max()),
    }
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
