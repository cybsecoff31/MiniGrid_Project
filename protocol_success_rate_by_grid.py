"""Fig. 1 - protocol success rate by grid size, trained PPO vs. uniform random.

Reads runs/baseline_doorkey{5x5,6x6,8x8,16x16}_seed1/eval.json and writes
fig1_success.png next to this script.

    python protocol_success_rate_by_grid.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")            # write a file, no window needed
import matplotlib.pyplot as plt
import numpy as np

# Paths relative to this file, so it works from any working directory
ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(ROOT, "runs")
OUT = os.path.join(ROOT, "fig1_success.png")

sizes = ["5x5", "6x6", "8x8", "16x16"]
ev = {}
for g in sizes:
    path = os.path.join(RUNS, f"baseline_doorkey{g}_seed1", "eval.json")
    with open(path) as f:
        ev[g] = json.load(f)

t = [ev[g]["trained"]["success_rate"] * 100 for g in sizes]
r = [ev[g]["random"]["success_rate"] * 100 for g in sizes]

fig, ax = plt.subplots(figsize=(5.2, 2.6))
x = np.arange(len(sizes))
w = 0.36
b1 = ax.bar(x - w / 2, t, w, color="#2a78d6", label="Trained PPO")
b2 = ax.bar(x + w / 2, r, w, color="#8a8986", label="Uniform random")

for bars in (b1, b2):                      # value label on every bar
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2,
                f"{b.get_height():.1f}%", ha="center", fontsize=7.5)

ax.set_xticks(x, [g.replace("x", "×") for g in sizes])
ax.set_xlabel("DoorKey grid size")
ax.set_ylabel("Success rate (%)")
ax.set_ylim(0, 112)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#e4e3df", linewidth=0.6)
ax.set_axisbelow(True)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)

fig.tight_layout()
fig.savefig(OUT, dpi=200)
print("wrote", OUT)
for g, a, b in zip(sizes, t, r):
    print(f"{g:>6}: trained {a:5.1f}%   random {b:5.1f}%")
