"""Compare two run directories and say whether they are bit-identical.

Used to check the reproducibility claim in the README: two runs with the same
seed on the same machine should produce the same metrics.csv.

    python compare_runs.py runs/<run_a> runs/<run_b>

Exit code 0 if the metric logs match, 1 if they do not.
"""

import csv
import json
import os
import sys

# wall clock and derived rates are expected to differ between runs
IGNORED = ("wall_time_s", "sps")


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def flatten(d, prefix=""):
    """Nested dict to dotted keys, so ppo.ent_coef reads as one line."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def show(v):
    return "(absent)" if v is None else json.dumps(v)


def table(header, rows):
    """Print rows as aligned columns under a header, two spaces indented."""
    widths = [max(len(str(c)) for c in col) for col in zip(header, *rows)]
    for row in (header, *rows):
        print("  " + "  ".join(str(c).ljust(w) for c, w in zip(row, widths)).rstrip())


def compare_config(a_dir, b_dir):
    with open(os.path.join(a_dir, "config.json")) as f:
        a = flatten(json.load(f))
    with open(os.path.join(b_dir, "config.json")) as f:
        b = flatten(json.load(f))
    a.pop("run_name", None)
    b.pop("run_name", None)
    keys = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    if not keys:
        print("config.json: identical")
        return True
    # Two runs with different configs are not a reproducibility test at
    # all, so this is a failure rather than a warning.
    print(f"config.json: {len(keys)} values differ, so this is not a same-config comparison")
    table(("key", "A", "B"), [(k, show(a.get(k)), show(b.get(k))) for k in keys])
    return False


def compare_csv(name, a_dir, b_dir):
    a = read_csv(os.path.join(a_dir, name))
    b = read_csv(os.path.join(b_dir, name))
    if len(a) != len(b):
        print(f"{name}: different number of rows, {len(a)} vs {len(b)}")
        return False
    bad = []
    for i, (ra, rb) in enumerate(zip(a, b)):
        for k in ra:
            if k not in IGNORED and ra[k] != rb[k]:
                bad.append((i, k, ra[k], rb[k]))
    if bad:
        print(f"{name}: {len(bad)} cells differ, first {min(10, len(bad))} shown")
        table(("row", "field", "A", "B"), bad[:10])
        return False
    print(f"{name}: identical, {len(a)} rows ({' and '.join(IGNORED)} ignored)")
    return True


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    a_dir, b_dir = sys.argv[1], sys.argv[2]
    print(f"A = {a_dir}\nB = {b_dir}\n")
    ok = compare_config(a_dir, b_dir)
    for name in ("metrics.csv", "episodes.csv"):
        ok = compare_csv(name, a_dir, b_dir) and ok
    print("\nRESULT: runs match" if ok else "\nRESULT: runs DO NOT match")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
