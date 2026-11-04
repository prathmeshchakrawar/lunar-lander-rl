import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make("LunarLander-v3")                  # raw game, no extra rewards
model = PPO.load("models/baseline", device="cpu")
mean, std = evaluate_policy(model, env, n_eval_episodes=20, deterministic=True)
print(f"baseline: {mean:.1f} +/- {std:.1f}   (solved = 200+)")