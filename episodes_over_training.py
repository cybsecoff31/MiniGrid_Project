"""Fig. 5 - every training episode of one run: its length and whether it reached the goal.

Reads runs/<run>/episodes.csv (one row per finished training episode, written by
ppo_minigrid.py: global_step, episodic_return, episodic_length) and writes
fig4_episodes8x8.png next to this script. Also prints the numbers quoted in
Report section 2.

    python episodes_over_training.py
    python episodes_over_training.py --run runs/cmp_shaped_8x8_seed1 --out fig5_shaped_seed1.png

Notes
  * x = global_step at which the episode ENDED, summed over all parallel envs.
  * success = episodic_return > 0 (same rule as the evaluation protocol).
  * The time limit is taken as the longest episode in the file; for DoorKey it
    equals MiniGrid's max_steps = 10 * N * N (640 for 8x8).
  * The "last success" line and the legend counts are computed from the data.
  * For a Group 2 run, episodes.csv logs the SPARSE return (statistics are
    recorded before the milestone bonus), so "success" still means reaching the goal.
Needs only numpy and matplotlib.
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=os.path.join(ROOT, "runs", "baseline_doorkey8x8_seed1"))
    ap.add_argument("--out", default=os.path.join(ROOT, "fig4_episodes8x8.png"))
    ap.add_argument("--random-rate", type=float, default=0.015,
                    help="random-policy success rate used for the chance comparison (from eval.json)")
    args = ap.parse_args()

    with open(os.path.join(args.run, "episodes.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    gs = np.array([int(r["global_step"]) for r in rows])
    L = np.array([int(r["episodic_length"]) for r in rows])
    R = np.array([float(r["episodic_return"]) for r in rows])
    succ = R > 0
    cap = int(L.max())

    # ---- numbers quoted in the report ----
    print(f"run: {args.run}")
    print(f"finished episodes:        {len(rows):,}")
    print(f"time limit (longest ep.): {cap}")
    print(f"episodes at the limit:    {(L == cap).sum():,} ({(L == cap).mean()*100:.1f}%)")
    print(f"successes (return > 0):   {succ.sum()}")
    if succ.any():
        last = int(gs[succ].max())
        before = gs <= last
        after = ~before
        n_after = int(after.sum())
        print(f"success steps:            {', '.join(f'{x:,}' for x in gs[succ])}")
        print(f"success lengths:          {', '.join(str(x) for x in L[succ])}")
        print(f"success returns:          {', '.join(f'{x:.3f}' for x in R[succ])}")
        print(f"last success at step:     {last:,}")
        print(f"rate up to last success:  {succ[before].sum()}/{before.sum()} = {succ[before].mean()*100:.1f}%")
        print(f"episodes after it:        {n_after:,}, successes {succ[after].sum()}")
        for name, p in (("random", args.random_rate), ("early", succ[before].mean())):
            print(f"  at the {name} rate {p*100:.1f}%: expected {n_after*p:.1f}, "
                  f"P(0 | independent) = {(1-p)**n_after:.1e}")

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(5.2, 2.4))
    k = gs / 1e3
    ax.scatter(k[~succ], L[~succ], s=5, color="#8a8986", alpha=0.5, linewidths=0,
               label=f"Timed out ({(~succ).sum():,})")
    ax.scatter(k[succ], L[succ], s=36, color="#eb6834", edgecolors="white", linewidths=1.5, zorder=3,
               label=f"Reached goal ({succ.sum()})")
    if succ.any():
        last_k = k[succ].max()
        ax.axvline(last_k, color="#52514e", lw=0.8, ls=":")
        ax.text(last_k + 5, cap * 0.12, f"last success\nstep {int(gs[succ].max()):,}", fontsize=7, color="#52514e")
    ax.set_xlabel("Environment steps (thousands)")
    ax.set_ylabel("Episode length")
    ax.set_ylim(0, cap * 1.08)
    ax.set_title(f"Every training episode ({len(rows):,})", fontsize=9, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#e4e3df", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=7.5, frameon=False)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
