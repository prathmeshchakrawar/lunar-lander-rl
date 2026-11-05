import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO, SAC

MODELS = [
    ("PPO discrete", PPO, "baseline", False),
    ("SAC continuous", SAC, "sac_base", True),
]
WINDS = [0, 5, 10, 15]

def run(model, continuous, wind, n=20):
    env = gym.make("LunarLander-v3", continuous=continuous,
                   enable_wind=wind > 0, wind_power=float(wind),
                   turbulence_power=wind / 10)
    scores, lengths = [], []
    for ep in range(n):
        obs, _ = env.reset(seed=ep)
        total, steps, done = 0.0, 0, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(action)
            total += r
            steps += 1
            done = term or trunc
        scores.append(total)
        lengths.append(steps)
    return np.mean(scores), np.mean(lengths)

print(f"{'model':16} {'wind':>5} {'score':>8} {'length':>8}")
for label, cls, name, cont in MODELS:
    model = cls.load(f"models/{name}", device="cpu")
    for w in WINDS:
        s, l = run(model, cont, w)
        print(f"{label:16} {w:>5} {s:>8.1f} {l:>8.0f}")