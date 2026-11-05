import sys
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO

name = sys.argv[1]
env = gym.make("LunarLander-v3", render_mode="rgb_array")
env = RecordVideo(env, video_folder="videos", name_prefix=f"stall_{name}",
                  episode_trigger=lambda ep: True)
model = PPO.load(f"models/{name}", device="cpu")

for ep in range(8):
    obs, _ = env.reset(seed=ep)
    steps, done = 0, False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, _ = env.step(action)
        steps += 1
        done = term or trunc
    print(f"episode {ep}: {steps} frames, timeout={trunc}")
env.close()