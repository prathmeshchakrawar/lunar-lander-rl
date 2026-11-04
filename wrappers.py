import gymnasium as gym
import numpy as np


class TimeFee(gym.Wrapper):
    """Small charge every frame so waiting costs something."""
    def __init__(self, env, fee=0.02):
        super().__init__(env)
        self.fee = fee

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        return obs, r - self.fee, term, trunc, info


class HoverBonus(gym.Wrapper):
    """The deliberately BAD reward: +1 per frame for being upright and slow."""
    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        x, y, vx, vy, angle, spin, left, right = obs
        if abs(angle) < 0.2 and np.hypot(vx, vy) < 0.3:
            r += 1.0
        return obs, r, term, trunc, info


class ProgressFix(gym.Wrapper):
    """The fix: pay for IMPROVEMENT (change in potential), plus a time fee."""
    def __init__(self, env, gamma=0.999, fee=0.02):
        super().__init__(env)
        self.gamma, self.fee, self.prev_phi = gamma, fee, 0.0

    @staticmethod
    def _phi(obs):
        return -float(np.hypot(obs[0], obs[1]))     # closer to pad = higher

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_phi = self._phi(obs)
        return obs, info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        phi_next = 0.0 if term else self._phi(obs)
        r += self.gamma * phi_next - self.prev_phi - self.fee
        self.prev_phi = phi_next
        return obs, r, term, trunc, info


WRAPPERS = {"none": None, "time": TimeFee, "hack": HoverBonus, "fix": ProgressFix}