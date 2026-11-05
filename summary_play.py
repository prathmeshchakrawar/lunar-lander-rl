import csv
from collections import defaultdict
import numpy as np

by_mode = defaultdict(list)
for mode, score in csv.reader(open("play_scores.csv")):
    by_mode[mode].append(float(score))

for mode, s in by_mode.items():
    s = np.array(s)
    std = s.std(ddof=1) if len(s) > 1 else 0.0
    print(f"{mode:10} n={len(s):2d}  mean {s.mean():7.1f}  std {std:6.1f}  "
          f"landed (>=200): {(s >= 200).sum()}/{len(s)}")