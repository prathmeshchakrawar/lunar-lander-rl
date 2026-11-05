import argparse
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env

p = argparse.ArgumentParser()
p.add_argument("--name", default="sac_base")
p.add_argument("--steps", type=int, default=500_000)
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

env = make_vec_env("LunarLander-v3", n_envs=1, seed=args.seed,
                   env_kwargs=dict(continuous=True))

model = SAC("MlpPolicy", env, device="cpu",
            learning_rate=7e-4, buffer_size=1_000_000, batch_size=256,
            gamma=0.99, tau=0.01, learning_starts=10_000,
            policy_kwargs=dict(net_arch=[256, 256]),
            tensorboard_log="logs", seed=args.seed, verbose=1)
model.learn(total_timesteps=args.steps)
model.save(f"models/{args.name}")
print("saved to", f"models/{args.name}")