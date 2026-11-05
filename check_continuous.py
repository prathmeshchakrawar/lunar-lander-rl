import gymnasium as gym

env = gym.make("LunarLander-v3", continuous=True)
print("action space:", env.action_space)
print("observation space:", env.observation_space.shape)

obs, _ = env.reset(seed=0)
for _ in range(3):
    a = env.action_space.sample()
    print("sample action:", a.round(2))