import pygame
import gymnasium as gym
from stable_baselines3 import SAC

env = gym.make("LunarLander-v3", continuous=True, render_mode="human")
model = SAC.load("models/sac_base", device="cpu")

obs, _ = env.reset()
ai_mode, total = False, 0.0            # start in MANUAL
prev_toggle = prev_r = False

print("Click the game window first. SPACE or TAB = switch AI/manual, R = reset, ESC = quit")

while True:
    pygame.event.pump()
    if pygame.event.get(pygame.QUIT):
        break
    keys = pygame.key.get_pressed()
    if keys[pygame.K_ESCAPE]:
        break

    toggle = keys[pygame.K_SPACE] or keys[pygame.K_TAB]
    r_key = keys[pygame.K_r]
    if toggle and not prev_toggle:
        ai_mode = not ai_mode
        print("mode ->", "AI (SAC)" if ai_mode else "MANUAL")
    if r_key and not prev_r:
        obs, _ = env.reset()
        total = 0.0
        print("reset")
    prev_toggle, prev_r = toggle, r_key

    if ai_mode:
        action, _ = model.predict(obs, deterministic=True)
    else:
        main = 1.0 if (keys[pygame.K_UP] or keys[pygame.K_w]) else -1.0   # < 0 = engine off
        side = 0.0                                                         # |side| < 0.5 = off
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            side = 1.0
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            side = -1.0
        action = [main, side]

    obs, reward, terminated, truncated, _ = env.step(action)
    total += reward

    mode = "AI (SAC)" if ai_mode else "MANUAL"
    pygame.display.set_caption(f"{mode} | return {total:.0f} | SPACE switch, R reset, ESC quit")

    if terminated or truncated:
        print(f"{mode}: episode return {total:.1f}")
        open("play_scores.csv", "a").write(f"{mode},{total:.1f}\n")
        obs, _ = env.reset()
        total = 0.0

env.close()