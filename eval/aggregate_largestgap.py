"""Aggregate per-width gap-sweep JSONs into one table per policy and report the
50%-success threshold (largest jumpable gap) for both crossing_rate and the
strict landed_on_platform_rate, via linear interpolation on the falling edge.
"""
import glob, json, os

OUT = "/root/vast/keming/SalkResearch/eval/gap_sweep_out"


def load_tag(tag):
    rows = []
    for f in sorted(glob.glob(os.path.join(OUT, f"{tag}_w*.json"))):
        rows.extend(json.load(open(f)))
    rows.sort(key=lambda r: r["gap_width_cm"])
    return rows


def thresh50(rows, key):
    xs = [r["gap_width_cm"] for r in rows]
    ys = [r[key] for r in rows]
    cross = None
    for i in range(1, len(xs)):
        if ys[i - 1] >= 0.5 and ys[i] < 0.5:
            x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
            cross = x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0)
    if cross is None:
        cross = f">={xs[-1]}" if ys[-1] >= 0.5 else f"<{xs[0]}"
    return cross


for tag in ("spd09_2w", "spd12_2w"):
    rows = load_tag(tag)
    if not rows:
        print(f"[{tag}] no results yet"); continue
    print(f"\n===== {tag}  ({len(rows)} widths) =====")
    print(f"{'gap_cm':>6} {'cross':>6} {'landed':>7} {'safe_stop':>9} {'fell':>6} {'land_off_cm':>11}")
    for r in rows:
        off = r["mean_landing_offset_cm"]
        print(f"{r['gap_width_cm']:>6.0f} {r['crossing_rate']:>6.2f} "
              f"{r['landed_on_platform_rate']:>7.2f} {r['safe_stop_rate']:>9.2f} "
              f"{r['fell_rate']:>6.2f} {('%.1f' % off) if off is not None else 'NA':>11}")
    print(f"  50% threshold (largest jumpable gap):")
    print(f"     crossing_rate         -> {thresh50(rows, 'crossing_rate')} cm")
    print(f"     landed_on_platform    -> {thresh50(rows, 'landed_on_platform_rate')} cm")
