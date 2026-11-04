import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO

env = gym.make("LunarLander-v3", render_mode="rgb_array")
env = RecordVideo(env, video_folder="videos", name_prefix="baseline",
                  episode_trigger=lambda ep: True)
model = PPO.load("models/baseline", device="cpu")

for _ in range(3):
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, _ = env.step(action)
        done = term or trunc
env.close()
print("videos saved")