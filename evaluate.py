import sys
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

name = sys.argv[1]
env = gym.make("LunarLander-v3")                  # raw game, no wrapper
model = PPO.load(f"models/{name}", device="cpu")
mean, std = evaluate_policy(model, env, n_eval_episodes=20, deterministic=True)
print(f"{name}: {mean:.1f} +/- {std:.1f}   (solved = 200+)")