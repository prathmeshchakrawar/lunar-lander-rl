import sys
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC

name = sys.argv[1]
env = gym.make("LunarLander-v3", continuous=True)    # raw game, no wrapper
model = SAC.load(f"models/{name}", device="cpu")

scores, lengths, timeouts = [], [], 0
for ep in range(20):
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
    timeouts += int(trunc)

print(f"{name}: score {np.mean(scores):.1f} +/- {np.std(scores):.1f} | "
      f"avg length {np.mean(lengths):.0f} | timeouts {timeouts}/20")