import sys
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import SAC

name = sys.argv[1]
env = gym.make("LunarLander-v3", continuous=True, render_mode="rgb_array")
env = RecordVideo(env, video_folder="videos", name_prefix=name,
                  episode_trigger=lambda ep: True)
model = SAC.load(f"models/{name}", device="cpu")

for ep in range(3):
    obs, _ = env.reset(seed=ep)
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, _ = env.step(action)
        done = term or trunc
env.close()
print("videos saved")