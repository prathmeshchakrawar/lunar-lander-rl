import csv
import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO

# seed 0 models have the old names; seeds 1 and 2 use the _s1 / _s2 names
CONFIGS = {
    "baseline": ["baseline", "none_s1", "none_s2"],
    "time_fee": ["time_fee", "time_s1", "time_s2"],
    "hack":     ["hack", "hack_s1", "hack_s2"],
    "fix":      ["fix", "fix_s1", "fix_s2"],
}

def evaluate(name, n_episodes=20):
    env = gym.make("LunarLander-v3")                    # raw game, no wrapper
    model = PPO.load(f"models/{name}", device="cpu")
    scores, lengths, timeouts = [], [], 0
    for ep in range(n_episodes):
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
    return np.mean(scores), np.mean(lengths), timeouts / n_episodes

rows = []
print(f"{'config':10} {'seeds':>5} {'score mean +/- std':>22} {'length':>8} {'timeout%':>9}")
for cfg, names in CONFIGS.items():
    found = [n for n in names if os.path.exists(f"models/{n}.zip")]
    res = [evaluate(n) for n in found]
    s = np.array([r[0] for r in res])      # one score per seed
    l = np.array([r[1] for r in res])
    t = np.array([r[2] for r in res])
    std = s.std(ddof=1) if len(s) > 1 else 0.0
    print(f"{cfg:10} {len(found):>5} {s.mean():>13.1f} +/- {std:<6.1f} {l.mean():>8.0f} {t.mean()*100:>8.0f}%")
    rows.append([cfg, len(found), round(s.mean(), 1), round(std, 1),
                 round(l.mean()), round(t.mean() * 100)])

with open("results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["config", "n_seeds", "score_mean", "score_std", "avg_length", "timeout_pct"])
    w.writerows(rows)
print("saved results.csv")