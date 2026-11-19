#!/usr/bin/env python3
"""Dilution-aware GTA VI trophy rarity forecaster (Python 3.13, stdlib only).

Model, fitted at run time from the case file itself:

Exophase does not refresh a game's counts continuously: between refreshes
the page is frozen (same players tracked, same rates), and at a refresh the
players-tracked count jumps while almost every rate falls, because the new
players have not earned the trophies yet.  A forecast that says "rates stay
put" therefore misses every refresh.  This program models the two quantities
a rate is made of -- earners and players -- separately, in log time:

1.  Players.  log(players_tracked) is fitted against ln(1 + hours since
    launch, 2026-11-19T00:00Z) with a Theil-Sen slope over the distinct
    refresh points of the history (a read only counts when its players count
    or rates differ from the previous read), shrunk halfway
    toward a slope of 1 (players grow roughly linearly with time since
    launch while a game is new).  When the history is completely flat -- no
    refresh yet, as on launch night -- the slope prior of 1 is used,
    anchored at the first read: the frozen count is stale since the page
    last changed, so the target count is extrapolated from when that count
    was fresh, not from the checkpoint.
2.  Earners per trophy.  E = rate * players / 100 at each refresh point.
    log(E) is fitted the same way, per trophy, with a Theil-Sen slope
    shrunk halfway toward the median slope of its tier (story trophies
    early, platinum and rare ones late; a single trophy's slope over a
    handful of refreshes is noisy).  When the history is flat the slope
    prior is tier-based (bronze 0.15, silver 0.10, gold 0.05, platinum 0):
    earners of already-open trophies keep creeping up while the player
    base jumps, so the rate still falls, just less than pure dilution.
    A trophy every tracked player holds (rate 100%) keeps rate 100%.
3.  Median rate = 100 * E_target / P_target.
4.  Spread.  Quantiles are logistic(logit(median) + z_tau * sigma) with
    sigma = sqrt(0.8^2 + (0.5*sqrt(days))^2 + (0.35*ln(P_target/P_last))^2
    + resid^2): a floor for refresh risk, a random-walk term growing with
    the horizon, a dilution-uncertainty term growing with how far the
    player count has to jump, and the trophy's own fit residual.  The band
    is deliberately wide -- a refresh can shift every rate at once, so a
    narrow band pays on every trophy at once.

Output is exactly one row per trophy, five non-decreasing quantiles in
[0.01, 99.99] (100% trophies keep q50 = 100), 4 decimals, deterministic:
no randomness anywhere, so two runs write identical bytes.

Run: python3 main.py, reading $MAKEFUN_INPUT_DIR (else /input/) and
writing $MAKEFUN_OUTPUT_DIR (else /output/), one .out per .in file.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

LAUNCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)
Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)
# Slope prior for earners growth when a trophy's history is flat, by tier.
TIER_PRIOR = {"bronze": 0.15, "silver": 0.10, "gold": 0.05, "platinum": 0.0}
SIGMA_FLOOR = 0.8
SIGMA_RW = 0.5
SIGMA_DILUTION = 0.35


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_launch(dt):
    return (dt - LAUNCH).total_seconds() / 3600.0


def log_time(dt):
    # ln(1 + hours since launch); the page's first reads are just before
    # launch, so 1 + hours stays positive (clamped for safety).
    return math.log(max(1.0 + hours_since_launch(dt), 0.05))


def median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def theil_sen(xs, ys):
    """Median pairwise slope; None when there is no usable pair."""
    slopes = []
    n = len(xs)
    for i in range(n):
        xi, yi = xs[i], ys[i]
        for j in range(i + 1, n):
            dx = xs[j] - xi
            if dx > 1e-9:
                slopes.append((ys[j] - yi) / dx)
    if not slopes:
        return None
    return median(slopes)


def fit_log_curve(xs, ys, prior_slope, shrink=0.5):
    """Fit log y = a + k * x with a Theil-Sen slope shrunk toward a prior.

    Returns (intercept, slope, residual_std).  With fewer than 3 distinct
    x values, or no variation in y, the prior slope is used and the
    intercept is anchored at the last point.
    """
    distinct_x = len(set(round(v, 9) for v in xs))
    varied_y = (max(ys) - min(ys)) > 1e-9 if ys else False
    slope = None
    if distinct_x >= 3 and varied_y and len(xs) >= 3:
        slope = theil_sen(xs, ys)
    if slope is None:
        slope = prior_slope
    else:
        slope = (1.0 - shrink) * slope + shrink * prior_slope
    slope = max(-0.5, min(2.0, slope))
    # Intercept as the median residual level (robust to an odd refresh).
    intercept = median([y - slope * x for x, y in zip(xs, ys)]) if xs else 0.0
    if xs:
        r = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
        resid_std = math.sqrt(sum(v * v for v in r) / len(r)) if r else 0.0
    else:
        resid_std = 0.0
    return intercept, slope, min(resid_std, 0.5)


def refresh_points(history):
    """One point per refresh: reads where players_tracked or any rate
    differs from the previous read.  Between refreshes the page is
    byte-frozen, so repeated reads carry no new information; keeping them
    would overweight the stale segment and drag a fitted line down (a
    frozen count must be extrapolated from when it was fresh -- its first
    read -- not from the checkpoint)."""
    pts = []
    prev_p = None
    prev_rates = None
    for read in history:
        p = read.get("players_tracked")
        if p is None:
            continue
        rates = tuple(read.get("rates") or [])
        if p != prev_p or rates != prev_rates:
            pts.append(read)
            prev_p = p
            prev_rates = rates
    return pts


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def logistic(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x)) if x < 700 else 1.0
    ex = math.exp(x)
    return ex / (1.0 + ex)


def logit(p):
    p = clamp(p, 0.0001, 0.9999)
    return math.log(p / (1.0 - p))


def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history") or []
    horizon_min = float(case.get("horizon_minutes") or 0)
    try:
        target = parse_time(case["target"])
    except Exception:
        target = parse_time(case["at"])
    x_target = log_time(target)
    days = max(horizon_min, 0.0) / 1440.0

    pts = refresh_points(history)
    # Cap fit points for speed (final history can hold hundreds of reads;
    # refreshes are far fewer, but stay safe).
    if len(pts) > 80:
        step = len(pts) / 80.0
        pts = [pts[int(i * step)] for i in range(80)]
        if history and pts[-1] is not history[-1]:
            pts.append(history[-1])

    # ---- players -------------------------------------------------------
    p_last = None
    if history:
        for read in reversed(history):
            if read.get("players_tracked"):
                p_last = float(read["players_tracked"])
                break
    if pts and p_last is not None:
        xs = [log_time(parse_time(r["at"])) for r in pts]
        ys = [math.log(max(float(r["players_tracked"]), 1.0)) for r in pts]
        a_p, k_p, _ = fit_log_curve(xs, ys, prior_slope=1.0, shrink=0.5)
        p_target = math.exp(a_p + k_p * x_target)
    else:
        p_target = p_last or 1.0
    p_target = clamp(p_target, p_last or 1.0, (p_last or 1.0) * 200.0)
    log_growth = math.log(max(p_target / (p_last or 1.0), 1.0))

    # ---- earners, pass 1: raw slopes ----------------------------------
    n = len(trophies)
    raw_slopes = [None] * n
    intercepts = [0.0] * n
    resid_stds = [0.0] * n
    e_last = [0.0] * n
    always_full = [False] * n
    for i, tr in enumerate(trophies):
        rates = []
        for r in pts:
            rs = r.get("rates") or []
            if i < len(rs) and rs[i] is not None:
                rates.append((r, float(rs[i])))
        full = rates and all(v >= 99.995 for _, v in rates)
        always_full[i] = bool(full)
        if rates:
            e_last[i] = rates[-1][1] * (p_last or 1.0) / 100.0
        if full or not rates:
            continue
        xs = [log_time(parse_time(r["at"])) for r, _ in rates]
        ys = [math.log(max(v * float(r["players_tracked"]) / 100.0, 0.5))
              for r, v in rates]
        if len(xs) >= 3 and (max(ys) - min(ys)) > 1e-9:
            s = theil_sen(xs, ys)
            if s is not None:
                raw_slopes[i] = max(-0.5, min(2.0, s))

    # ---- tier median slopes for shrinkage ------------------------------
    tier_slopes = {}
    for tier in ("bronze", "silver", "gold", "platinum"):
        vals = [raw_slopes[i] for i, tr in enumerate(trophies)
                if tr.get("tier") == tier and raw_slopes[i] is not None]
        if vals:
            tier_slopes[tier] = median(vals)
    all_raw = [s for s in raw_slopes if s is not None]
    global_med = median(all_raw) if all_raw else 0.1

    # ---- earners, pass 2: shrunk fits, medians, spreads ---------------
    rows = []
    for i, tr in enumerate(trophies):
        tid = tr["id"]
        if always_full[i]:
            rows.append({"id": tid, "q": [99.0, 99.8, 100.0, 100.0, 100.0]})
            continue
        tier = tr.get("tier") or "bronze"
        prior = TIER_PRIOR.get(tier, 0.15)
        rates = []
        for r in pts:
            rs = r.get("rates") or []
            if i < len(rs) and rs[i] is not None:
                rates.append((r, float(rs[i])))
        if rates:
            xs = [log_time(parse_time(r["at"])) for r, _ in rates]
            ys = [math.log(max(v * float(r["players_tracked"]) / 100.0, 0.5))
                  for r, v in rates]
            if raw_slopes[i] is not None:
                centre = tier_slopes.get(tier, global_med)
                slope = 0.5 * raw_slopes[i] + 0.5 * centre
                slope = max(-0.5, min(2.0, slope))
            else:
                slope = prior
            intercept = median([y - slope * x for x, y in zip(xs, ys)])
            r = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
            resid = math.sqrt(sum(v * v for v in r) / len(r)) if r else 0.0
            e_target = math.exp(intercept + slope * x_target)
        else:
            slope = prior
            resid = 0.0
            e_target = max(e_last[i], 0.05)
        e_target = max(e_target, 0.05)
        med = 100.0 * e_target / p_target
        med = clamp(med, 0.01, 99.99)
        sigma = math.sqrt(SIGMA_FLOOR ** 2
                          + (SIGMA_RW * math.sqrt(days)) ** 2
                          + (SIGMA_DILUTION * log_growth) ** 2
                          + min(resid, 0.5) ** 2)
        sigma = min(sigma, 2.5)
        base = logit(med / 100.0)
        qs = []
        prev = -1.0
        for z in Z:
            q = round(100.0 * logistic(base + z * sigma), 4)
            q = clamp(q, 0.01, 99.99)
            if q < prev:
                q = prev
            qs.append(q)
            prev = q
        rows.append({"id": tid, "q": qs})
    rows.sort(key=lambda row: row["id"])
    return {"trophies": rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input/"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output/"
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(fn for fn in os.listdir(in_dir) if fn.endswith(".in"))
    for name in names:
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as f:
            case = json.load(f)
        out = forecast_case(case)
        stem = name[:-3] if name.endswith(".in") else name
        with open(os.path.join(out_dir, stem + ".out"), "w",
                  encoding="utf-8") as f:
            f.write(json.dumps(out, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
