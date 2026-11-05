# Lunar Lander: reward shaping and reward hacking

Trained PPO agents (Stable-Baselines3) on Gymnasium's `LunarLander-v3`, then
deliberately broke the reward to reproduce **reward hacking**, and fixed it.

## What I did
1. Trained a baseline PPO agent (1M steps) that lands the rocket.
2. Added a **time fee** (-0.02 per frame).
3. Added a deliberately bad **hover bonus** (+1 per frame when upright and slow).
4. Replaced it with a **progress-based reward** (change in potential, plus a time fee).

## Results (3 seeds, 20 evaluation episodes each, wrapper OFF)

| Config | Score (mean ± std across seeds) | Avg episode length | Timeouts |
|---|---|---|---|
| Baseline | 251.1 ± 10.8 | 364 | 2% |
| Time fee | 261.4 ± 4.0 | 320 | 0% |
| Hover bonus (hack) | 36.5 ± 96.9 | 949 | 68% |
| Progress fix | 256.5 ± 14.0 | 337 | 2% |

Solved = 200+. Scores come from the raw environment so the custom bonus cannot inflate them.

## Findings
- The hover bonus made the agent stay in the air: average episode length went
  from 364 to 949 frames, and 68% of episodes ran to the 1000-frame limit.
  The agent optimised my reward, not the task.
- Scores for the hack varied a lot across seeds (std 96.9). A single seed
  (148 on seed 0) understated the effect, so one run is not enough.
- The progress-based reward removed the hover behaviour (score 256.5).
- Differences between baseline, time fee and fix are within one standard
  deviation with only 3 seeds, so I do not claim any of them is better.

## Videos
- `videos/baseline-episode-0.mp4`: lands normally
- `videos/hack_s1-episode-0.mp4`: hovers (replace with your clearest clip)
- `videos/fix-episode-0.mp4`: lands again

## How to run
```bash
pip install "gymnasium[box2d]" stable-baselines3 moviepy tensorboard
python train.py --reward none --name baseline      # none | time | hack | fix
python evaluate_len.py baseline
python record.py baseline
python results.py
```

## Files
- `wrappers.py`: reward wrappers (TimeFee, HoverBonus, ProgressFix)
- `train.py`, `evaluate_len.py`, `record.py`, `results.py`, `run_seeds.sh`