"""Aggregate eval.json across seeds into a per-group spread table.

    python aggregate_eval.py --group NAME run_dir run_dir ... [--group NAME ...]

For each group it reads every run's eval.json (the protocol result: 200 episodes,
seed 12345, sparse objective) and reports the mean and the spread ACROSS SEEDS of
the trained policy's success rate / return / length, plus the random-policy
reference. Reporting the spread, not just the mean, is a midterm requirement.
"""
import argparse
import json
import os
import statistics as st


def load(run_dir):
    with open(os.path.join(run_dir, "eval.json")) as f:
        return json.load(f)


def spread(xs):
    xs = list(xs)
    if len(xs) == 1:
        return xs[0], 0.0, xs[0], xs[0]
    return st.mean(xs), st.pstdev(xs), min(xs), max(xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", nargs="+", action="append", metavar="NAME RUN...",
                    required=True, help="group label followed by its run dirs")
    args = ap.parse_args()

    print(f"{'group':<18}{'seeds':>6}  {'success (mean/std/min/max)':<34}"
          f"{'return mean':>12}{'length mean':>13}{'rand succ':>11}")
    print("-" * 106)
    for group in args.group:
        name, dirs = group[0], group[1:]
        evs = []
        for d in dirs:
            if os.path.exists(os.path.join(d, "eval.json")):
                evs.append(load(d))
            else:
                print(f"  [missing] {d}/eval.json -- run not finished/evaluated yet")
        if not evs:
            print(f"{name:<18}{0:>6}  (no eval.json found)")
            continue
        succ = [e["trained"]["success_rate"] for e in evs]
        ret = [e["trained"]["return_mean"] for e in evs]
        ln = [e["trained"]["length_mean"] for e in evs]
        rand = [e["random"]["success_rate"] for e in evs if "random" in e]
        m, s, lo, hi = spread(succ)
        print(f"{name:<18}{len(evs):>6}  "
              f"{m:>5.2%} / {s:>5.2%} / {lo:>5.2%} / {hi:>5.2%}   "
              f"{st.mean(ret):>11.3f}{st.mean(ln):>13.1f}"
              f"{(st.mean(rand) if rand else float('nan')):>10.2%}")
    print("-" * 106)
    print("success = fraction of the 200 evaluation episodes with return > 0 "
          "(sparse objective), averaged over seeds; std is the population spread across seeds.")


if __name__ == "__main__":
    main()
