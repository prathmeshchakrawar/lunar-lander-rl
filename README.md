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

Solved = 200+. Scores come from the raw environment so the custom bonus cannot
inflate them. A timeout means the episode hit the 1000-frame limit without landing.

### Hack, per seed

| Seed | Score | Avg episode length | Timeouts (of 20) |
|---|---|---|---|
| 0 | 148.2 | 849 | 2 |
| 1 | -24.5 | 998 | 19 |
| 2 | -14.2 | 1000 | 20 |

## Findings
- The hover bonus made the agent stall: average episode length went from 364
  to 949 frames, and 68% of episodes hit the 1000-frame limit without landing.
  The agent optimised my reward, not the task.
- The effect depended on the seed. Two of three seeds almost never landed
  (19/20 and 20/20 episodes hit the 1000-frame limit, scores -24.5 and -14.2).
  The third landed in 18 of 20 episodes but only after about 849 frames on
  average, against 364 for the baseline. Same reward, different outcomes,
  which is why I report all three seeds.
- The progress-based reward removed the stalling (score 256.5, length 337).
- Differences between baseline, time fee and fix are within one standard
  deviation with only 3 seeds, so I do not claim any of them is better.

## Continuous control with SAC

Trained SAC with continuous actions (500k steps, one seed) on the same environment.
Evaluated over 20 episodes with the raw reward.

| Model | Score | Avg episode length |
|---|---|---|
| PPO, discrete actions (seed 0) | 250.0 | 382 |
| SAC, continuous actions (seed 0) | 282.5 | 187 |

### Robustness to wind (no retraining, trained without wind)

| Wind power | PPO discrete | SAC continuous |
|---|---|---|
| 0 | 250.0 | 282.5 |
| 5 | 241.6 | 282.6 |
| 10 | 192.3 | 261.2 |
| 15 | 145.3 | 255.4 |

Caveat: one seed per model. Both the algorithm (PPO vs SAC) and the action
space (discrete vs continuous) differ, so I cannot say which one causes the
gap. Next step is to repeat with more seeds.

## Videos
- `videos/baseline-episode-0.mp4`: lands normally
- `videos/stall_hack_s2-episode-0.mp4`: stall behaviour, the rocket drifts down very slowly and never lands within 1000 frames
- `videos/fix-episode-0.mp4`: lands again
- `videos/sac_base-episode-0.mp4`: SAC with continuous throttle

## How to run
```bash
pip install "gymnasium[box2d]" stable-baselines3 moviepy tensorboard
python train.py --reward none --name baseline      # none | time | hack | fix
python evaluate_len.py baseline
python record.py baseline
bash run_seeds.sh                                  # seeds 1 and 2 for all configs
python results.py
```

## Files
- `wrappers.py`: reward wrappers (TimeFee, HoverBonus, ProgressFix)
- `train.py`, `evaluate_len.py`, `record.py`, `record_stall.py`, `results.py`, `run_seeds.sh`
- `results.csv`: the table above
- `train_sac.py`, `evaluate_sac.py`, `record_sac.py`, `eval_wind.py`, `check_continuous.py`