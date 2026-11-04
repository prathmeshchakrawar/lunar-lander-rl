import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from wrappers import WRAPPERS

p = argparse.ArgumentParser()
p.add_argument("--reward", choices=WRAPPERS.keys(), default="none")
p.add_argument("--name", default="baseline")
p.add_argument("--steps", type=int, default=1_000_000)
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

env = make_vec_env("LunarLander-v3", n_envs=16, seed=args.seed,
                   wrapper_class=WRAPPERS[args.reward])

model = PPO("MlpPolicy", env, device="cpu",
            n_steps=1024, batch_size=64, n_epochs=4,
            gamma=0.999, gae_lambda=0.98, ent_coef=0.01,
            tensorboard_log="logs", seed=args.seed, verbose=1)
model.learn(total_timesteps=args.steps)
model.save(f"models/{args.name}")
print("saved to", f"models/{args.name}")