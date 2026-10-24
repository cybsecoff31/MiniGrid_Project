"""Fig. 2 - where the agent spends its time, decoded from the recorded evaluation GIFs.

Reads the GIFs written by
    python eval_checkpoint.py model_path=runs/baseline_doorkey8x8_seed1/model.pt \
        policy.also_random=true render.enabled=true
i.e. runs/baseline_doorkey8x8_seed1/render/{sampled,random}/ep000.gif ... ep004.gif,
and for every episode reconstructs, step by step:
    * the agent's cell            (tile containing the red triangle)
    * whether the key is still on the floor   (key tile still yellow)
    * whether the door is still locked        (door tile still olive-filled)

It prints a per-episode table (cells visited, share of steps on the most-visited
cell, step at which the key was picked up, step at which the door was unlocked)
and writes fig5_visits.png (cell-visit heat maps, trained vs. random) next to
this script.

    python cell_visits_from_gifs.py
    python cell_visits_from_gifs.py --run runs/baseline_doorkey8x8_seed1 --layouts 0 3

How the GIFs are decoded
    * eval_checkpoint.py saves GIFs with optimize=True, which MERGES identical
      consecutive frames and lengthens their duration. Each frame therefore stands
      for round(duration / frame_ms) steps (frame_ms = 200 by default, see
      conf/eval.yaml render.frame_ms). For an 8x8 timeout the durations add up to
      641 frames = reset frame + 640 steps.
    * Frames are drawn with render.tile_size = 32 px per cell, so a 256x256 frame
      is an 8x8 grid of tiles.
    * MiniGrid does not draw an object the agent is carrying, so "key tile no
      longer yellow" means the key was picked up (at least once).
    * Colour tests use colour DIFFERENCES, not fixed RGB values, because the
      shaded 7x7 view (render.highlight) lightens every tile it covers.
Needs only numpy, pillow and matplotlib.
"""
import argparse
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
TILE = 32        # conf/eval.yaml render.tile_size
FRAME_MS = 200   # conf/eval.yaml render.frame_ms


# --------------------------------------------------------------------------
# Pixel tests on one 32x32 tile (int array, shape (32, 32, 3))
# --------------------------------------------------------------------------
def n_yellow(t):   # key and door outline: R, G high, B much lower
    return int(((t[..., 0] > 200) & (t[..., 1] > 200) & (t[..., 0] - t[..., 2] > 100)).sum())


def n_olive(t):    # locked-door fill: R == G, mid brightness, B much lower
    return int(((abs(t[..., 0] - t[..., 1]) < 8) & (t[..., 0] > 100) & (t[..., 0] < 200)
                & (t[..., 0] - t[..., 2] > 60)).sum())


def n_red(t):      # agent triangle
    return int(((t[..., 0] > 200) & (t[..., 0] - t[..., 1] > 100)).sum())


def n_green(t):    # goal
    return int(((t[..., 1] > 200) & (t[..., 0] < 80)).sum())


def tile(a, x, y):
    return a[y * TILE:(y + 1) * TILE, x * TILE:(x + 1) * TILE]


# --------------------------------------------------------------------------
# Decode one GIF into a per-step sequence
# --------------------------------------------------------------------------
def frames(path):
    """Yield (frame as int RGB array, number of steps this frame stands for)."""
    im = Image.open(path)
    for i in range(im.n_frames):
        im.seek(i)
        n = max(1, round(im.info.get("duration", FRAME_MS) / FRAME_MS))
        yield np.asarray(im.convert("RGB")).astype(int), n


def decode(path):
    """Return (door, key, goal) cells and a list with one
    (agent_cell, key_on_floor, door_locked) tuple per step (index 0 = reset)."""
    seq, layout = [], None
    for a, n in frames(path):
        N = a.shape[0] // TILE
        if layout is None:  # find objects once, on the reset frame
            cand, goal = [], None
            for y in range(N):
                for x in range(N):
                    t = tile(a, x, y)
                    if n_yellow(t) > 15:
                        cand.append((n_yellow(t) + n_olive(t), (x, y)))
                    if n_green(t) > 300:
                        goal = (x, y)
            cand.sort(reverse=True)               # door has the most yellow+olive
            door = cand[0][1]
            key = cand[1][1] if len(cand) > 1 else None
            layout = (door, key, goal)
        door, key, goal = layout
        agent = None
        for y in range(N):
            for x in range(N):
                if n_red(tile(a, x, y)) > 40:
                    agent = (x, y)
        key_on = bool(n_yellow(tile(a, *key)) > 15) if key else None
        locked = bool(n_olive(tile(a, *door)) > 300)
        seq += [(agent, key_on, locked)] * n
    return layout, seq


def episode_stats(layout, seq):
    door, key, goal = layout
    pos = [p for p, _, _ in seq if p is not None]
    visits = Counter(pos)
    return {
        "steps": len(seq) - 1,
        "cells_visited": len(visits),
        "top_cell_share": max(visits.values()) / len(pos),
        "key_picked_step": next((i for i, (_, k, _) in enumerate(seq) if k is False), None),
        "door_unlocked_step": next((i for i, (_, _, l) in enumerate(seq) if l is False), None),
        "steps_past_door": sum(1 for p in pos if p[0] > door[0]),
        "visits": visits,
    }


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=os.path.join(ROOT, "runs", "baseline_doorkey8x8_seed1"))
    ap.add_argument("--episodes", type=int, default=5, help="GIFs per policy to decode (ep000..)")
    ap.add_argument("--layouts", type=int, nargs="+", default=[0, 3], help="layouts shown in the figure")
    ap.add_argument("--out", default=os.path.join(ROOT, "fig2_visits.png"))
    args = ap.parse_args()
    render = os.path.join(args.run, "render")

    # ---- per-episode table (the numbers quoted in Report §2) ----
    results = {}
    print(f"{'episode':<12}{'cells':>6}{'top cell':>10}{'key picked':>12}{'door unlocked':>15}{'past door':>11}")
    for pol in ("sampled", "random"):
        for ep in range(args.episodes):
            path = os.path.join(render, pol, f"ep{ep:03d}.gif")
            if not os.path.exists(path):
                print(f"  missing {path}")
                continue
            layout, seq = decode(path)
            st = episode_stats(layout, seq)
            results[(pol, ep)] = (layout, seq, st)
            name = f"{'trained' if pol == 'sampled' else 'random'} ep{ep}"
            ks = st["key_picked_step"]; ds = st["door_unlocked_step"]
            print(f"{name:<12}{st['cells_visited']:>6}{st['top_cell_share']*100:>9.0f}%"
                  f"{('step ' + str(ks)) if ks is not None else 'no':>12}"
                  f"{('step ' + str(ds)) if ds is not None else 'no':>15}"
                  f"{'yes' if st['steps_past_door'] else 'no':>11}")
    for pol, label in (("sampled", "trained"), ("random", "random")):
        sts = [v[2] for (p, _), v in results.items() if p == pol]
        if sts:
            print(f"{label:>8} mean: {np.mean([s['cells_visited'] for s in sts]):.1f} cells, "
                  f"{np.mean([s['top_cell_share'] for s in sts])*100:.0f}% on top cell, "
                  f"key {sum(s['key_picked_step'] is not None for s in sts)}/{len(sts)}, "
                  f"door {sum(s['door_unlocked_step'] is not None for s in sts)}/{len(sts)}")

    # ---- Fig. 5: heat maps, trained vs. random, for the chosen layouts ----
    panels = [(pol, ep) for ep in args.layouts for pol in ("sampled", "random") if (pol, ep) in results]
    fig, axs = plt.subplots(1, len(panels), figsize=(1.65 * len(panels), 1.95))
    axs = np.atleast_1d(axs)
    im = None
    for ax, (pol, ep) in zip(axs, panels):
        (door, key, goal), seq, st = results[(pol, ep)]
        n = len(seq)
        H = np.zeros((8, 8))
        for (x, y), c in st["visits"].items():
            H[y, x] = c / n * 100
        im = ax.imshow(H, cmap="Blues", vmin=0, vmax=50)
        for cell, lab, col in ((door, "D", "#eda100"), (key, "K", "#eda100"), (goal, "G", "#008300")):
            if cell is not None:
                ax.text(cell[0], cell[1], lab, ha="center", va="center", fontsize=7, color=col, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{'Trained' if pol == 'sampled' else 'Random'} · layout {ep}\n"
                     f"{st['cells_visited']} cells, max {st['top_cell_share']*100:.0f}%", fontsize=7.5)
    cb = fig.colorbar(im, ax=list(axs), fraction=0.02, pad=0.02)
    cb.set_label("% of 640 steps", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
