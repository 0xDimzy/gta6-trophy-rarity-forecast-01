#!/usr/bin/env python3
"""gta6-trophy-rarity-forecast-01 entry v2: backtest-weighted ensemble forecaster.

The case gives every Exophase read of the GTA VI PS5 trophy page before the
checkpoint: for each of 55 trophies an earn rate (percent of tracked players
who earned it) and the players-tracked count N. Between Exophase refreshes the
page is frozen, so the history collapses to a handful of distinct states.
Rates move for two reasons: earners E_i only ever grow, and N grows as new
players sync -- mostly diluting rates early, while cohorts maturing into late
trophies pushes some rates up. A persistence forecast misses both.

This program fits, at run time and per trophy, an ensemble of small models:

  persist      last rate (the house baseline's median)
  logit_ln     logit(rate) linear in ln(1 + hours since 2026-11-18T23:00Z)
  logit_ln24   the same on the last ~30 hours of states
  logit_sat    logit(rate) = A - B * exp(-x / tau), tau on a grid
  en_loglog    log earners and log players each linear in x (Theil-Sen)
  en_power     log earners linear in log players (dilution exponent)
  en_sat       earners saturating (Weibull-CDF shape), players saturating

Players N(t) itself is forecast by a backtest-weighted blend of a log-log
line and a saturating curve. Model weights come from an internal backtest:
each model is refit on the states before the last one (and before the
second-to-last) and scored on the held-out state, pooled across trophies
with a per-trophy adjustment shrunk toward the pooled weights. The spread
sigma on the logit scale combines the ensemble's backtest residual scale
grown to the forecast horizon, the disagreement between model medians, and
a Poisson term for trophies with few earners. Story-chapter medians are
made non-increasing along the chapter chain (a player cannot complete
chapter 3 without chapter 2) and the platinum median is capped by every
other trophy's.

With too few distinct states to backtest (the practice cases are one frozen
launch-night page), it falls back to dilution priors: players keep growing
from when the frozen count was fresh, earners grow at tier-prior rates.

Deterministic, standard library only. Reads $MAKEFUN_INPUT_DIR (else
/input/), writes one .out per .in to $MAKEFUN_OUTPUT_DIR (else /output/).
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

T0 = datetime(2026, 11, 18, 23, 0, 0, tzinfo=timezone.utc)
Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

TAU_GRID_X = (0.4, 0.8, 1.6, 3.2)          # logit_sat, in x = ln(1+t) units
SAT_TAU_H = (12.0, 24.0, 48.0, 96.0, 192.0, 384.0)   # Weibull scale, hours
SAT_P = (0.5, 0.75, 1.0, 1.5)              # Weibull shape
TIER_PRIOR = {"bronze": 0.15, "silver": 0.10, "gold": 0.05, "platinum": 0.0}
MODEL_NAMES = ("persist", "logit_ln", "logit_ln24", "logit_sat",
               "en_loglog", "en_power", "en_sat")
SIGMA_SCALE = 0.75


# --------------------------------------------------------------------------
# small numeric helpers
# --------------------------------------------------------------------------

def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def logit(p):
    p = clamp(p, 0.0001, 0.9999)
    return math.log(p / (1.0 - p))


def logistic(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x)) if x < 700 else 1.0
    ex = math.exp(x)
    return ex / (1.0 + ex)


def round4(x):
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if float(value).is_integer() else value


def parse_time(s):
    return datetime.fromisoformat(str(s).replace("Z", "+00:00"))


def hours(dt):
    return (dt - T0).total_seconds() / 3600.0


def x_of(t):
    return math.log(1.0 + max(t, 0.0))


def wls_line(xs, ys, ws):
    """Weighted least squares y = a + b x. Returns (a, b, weighted sse)."""
    sw = sum(ws)
    if sw <= 0 or len(xs) < 2:
        return (ys[-1] if ys else 0.0), 0.0, 0.0
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    if sxx < 1e-12:
        return my, 0.0, sum(w * (y - my) ** 2 for y, w in zip(ys, ws))
    b = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws)) / sxx
    a = my - b * mx
    sse = sum(w * (y - (a + b * x)) ** 2 for x, y, w in zip(xs, ys, ws))
    return a, b, sse


def theil_sen(xs, ys):
    slopes = []
    n = len(xs)
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[j] - xs[i]
            if dx > 1e-9:
                slopes.append((ys[j] - ys[i]) / dx)
    if not slopes:
        return None
    return median(slopes)


def weibull_g(t, tau, p):
    """Saturating basis 1 - exp(-(t/tau)^p), t in hours."""
    if t <= 0:
        return 0.0
    return 1.0 - math.exp(-((t / tau) ** p))


# --------------------------------------------------------------------------
# case -> distinct refresh states
# --------------------------------------------------------------------------

def build_states(case):
    """Collapse the read history to distinct Exophase states.

    A state is kept when players_tracked or any rate differs from the
    previous read. Each state: t (hours since T0), x = ln(1+t), N, T
    (trophies_earned, may be None), rates (list, may hold None), and
    earners per trophy (rate * N / 100, cleaned to non-decreasing).
    """
    history = case.get("history") or []
    n_tr = len(case.get("trophies") or [])
    states = []
    prev_key = None
    for read in history:
        n = read.get("players_tracked")
        if n is None:
            continue
        rates = read.get("rates") or []
        key = (n, tuple(rates))
        if key == prev_key:
            continue
        prev_key = key
        try:
            t = hours(parse_time(read["at"]))
        except Exception:
            continue
        rates = [float(r) if r is not None else None for r in rates]
        while len(rates) < n_tr:
            rates.append(None)
        states.append({
            "t": t, "x": x_of(t), "N": float(n),
            "T": float(read["trophies_earned"]) if read.get("trophies_earned") is not None else None,
            "rates": rates[:n_tr],
        })
    # earners, cleaned non-decreasing per trophy (counts never shrink)
    for i in range(n_tr):
        run_max = 0.0
        for st in states:
            r = st["rates"][i]
            e = (r * st["N"] / 100.0) if r is not None else None
            if e is not None:
                run_max = max(run_max, e)
                e = run_max
            st.setdefault("E", [None] * n_tr)[i] = e
    return states


# --------------------------------------------------------------------------
# players-tracked models (also reused for trophies_earned totals)
# --------------------------------------------------------------------------

def _fit_count_loglog(states, key):
    """log count linear in x. Returns predict(t) or None."""
    pts = [(st["x"], math.log(max(st[key], 1.0))) for st in states if st.get(key)]
    if len(pts) < 2:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ws = [0.5 + 0.5 * (i + 1) / len(pts) for i in range(len(pts))]
    a, b, _ = wls_line(xs, ys, ws)
    b = clamp(b, -0.2, 2.5)
    return lambda t: math.exp(a + b * x_of(t))


def _fit_count_sat(states, key):
    """count = C * (1 - exp(-(t/tau)^p)), grid over (tau, p)."""
    pts = [(st["t"], float(st[key])) for st in states if st.get(key)]
    if len(pts) < 3:
        return None
    best = None
    for tau in SAT_TAU_H:
        for p in SAT_P:
            gs = [weibull_g(t, tau, p) for t, _ in pts]
            denom = sum(g * g for g in gs)
            if denom < 1e-12:
                continue
            c = sum(g * y for g, (_, y) in zip(gs, pts)) / denom
            if c <= 0:
                continue
            sse = sum((y - c * g) ** 2 for g, (_, y) in zip(gs, pts))
            # relative sse so the choice is scale-free
            rel = sse / max(sum(y * y for _, y in pts), 1e-9)
            if best is None or rel < best[0]:
                best = (rel, c, tau, p)
    if best is None:
        return None
    _, c, tau, p = best
    return lambda t: c * weibull_g(t, tau, p)


def forecast_count(states, key, t_target, last_value):
    """Backtest-blended forecast of a count series (N or T) at t_target."""
    fits = {}
    f = _fit_count_loglog(states, key)
    if f:
        fits["loglog"] = f
    f = _fit_count_sat(states, key)
    if f:
        fits["sat"] = f
    if not fits:
        if len(states) == 1:
            # one frozen state: the count was fresh at that state's time;
            # grow it with log-log slope 1 from there (players roughly
            # linear in time since launch while a launch is young)
            t_fresh = states[0]["t"]
            return last_value * (1.0 + max(t_target, 0.0)) / (1.0 + max(t_fresh, 0.0))
        return last_value
    if len(fits) == 1 or len(states) < 4:
        # no backtest possible: average what we have (log space)
        vals = [math.log(max(f(t_target), 1e-9)) for f in fits.values()]
        return math.exp(sum(vals) / len(vals))
    # backtest: fit on states[:-1], predict last state's count
    errs = {}
    for name in fits:
        sub = states[:-1]
        f = _fit_count_loglog(sub, key) if name == "loglog" else _fit_count_sat(sub, key)
        actual = states[-1].get(key)
        if f is None or not actual:
            errs[name] = 1.0
        else:
            errs[name] = abs(math.log(max(f(states[-1]["t"]), 1e-9)) - math.log(actual))
    ws = {k: math.exp(-e / 0.10) for k, e in errs.items()}
    tot = sum(ws.values()) or 1.0
    val = sum((ws[k] / tot) * math.log(max(fits[k](t_target), 1e-9)) for k in fits)
    return math.exp(val)


# --------------------------------------------------------------------------
# per-trophy median models: each returns a rate in percent at t_target
# --------------------------------------------------------------------------

def _trophy_points(states, i):
    """(t, x, N, rate, E) rows where trophy i has a rate."""
    rows = []
    for st in states:
        r = st["rates"][i]
        if r is None:
            continue
        rows.append((st["t"], st["x"], st["N"], r, st["E"][i]))
    return rows


def _last_rate(states, i):
    for st in reversed(states):
        if st["rates"][i] is not None:
            return st["rates"][i]
    return None


def model_persist(states, i, t_target, n_target):
    r = _last_rate(states, i)
    return r if r is not None else 0.01


def _logit_line(states, i, t_target, recent_hours=None):
    rows = _trophy_points(states, i)
    if recent_hours is not None and rows:
        t_last = rows[-1][0]
        recent = [r for r in rows if r[0] >= t_last - recent_hours]
        if len(recent) >= 3:
            rows = recent
    if len(rows) < 2:
        return model_persist(states, i, t_target, None)
    xs = [r[1] for r in rows]
    ys = [logit(r[3] / 100.0) for r in rows]
    ws = [math.sqrt(max(r[2], 1.0)) for r in rows]
    a, b, _ = wls_line(xs, ys, ws)
    b = clamp(b, -3.0, 3.0)
    return 100.0 * logistic(a + b * x_of(t_target))


def model_logit_ln(states, i, t_target, n_target):
    return _logit_line(states, i, t_target)


def model_logit_ln24(states, i, t_target, n_target):
    return _logit_line(states, i, t_target, recent_hours=30.0)


def model_logit_sat(states, i, t_target, n_target):
    rows = _trophy_points(states, i)
    if len(rows) < 3:
        return model_persist(states, i, t_target, None)
    xs = [r[1] for r in rows]
    ys = [logit(r[3] / 100.0) for r in rows]
    ws = [math.sqrt(max(r[2], 1.0)) for r in rows]
    best = None
    for tau in TAU_GRID_X:
        gs = [-math.exp(-x / tau) for x in xs]
        a, b, sse = wls_line(gs, ys, ws)
        if best is None or sse < best[0]:
            best = (sse, a, b, tau)
    _, a, b, tau = best
    return 100.0 * logistic(a + b * (-math.exp(-x_of(t_target) / tau)))


def model_en_loglog(states, i, t_target, n_target):
    rows = [r for r in _trophy_points(states, i) if r[4] is not None]
    if len(rows) < 2 or not n_target:
        return model_persist(states, i, t_target, None)
    xs = [r[1] for r in rows]
    ys = [math.log(max(r[4], 0.25)) for r in rows]
    slope = theil_sen(xs, ys)
    if slope is None:
        slope = 0.0
    slope = clamp(slope, -0.3, 2.5)
    intercept = median([y - slope * x for x, y in zip(xs, ys)])
    e_target = math.exp(intercept + slope * x_of(t_target))
    e_target = max(e_target, rows[-1][4])  # earners never decrease
    return 100.0 * e_target / n_target


def model_en_power(states, i, t_target, n_target):
    rows = [r for r in _trophy_points(states, i) if r[4] is not None and r[2] > 0]
    if len(rows) < 2 or not n_target:
        return model_persist(states, i, t_target, None)
    xs = [math.log(r[2]) for r in rows]
    ys = [math.log(max(r[4], 0.25)) for r in rows]
    ws = [math.sqrt(max(r[4], 1.0)) for r in rows]
    a, b, _ = wls_line(xs, ys, ws)
    b = clamp(b, -0.3, 1.6)
    e_target = math.exp(a + b * math.log(n_target))
    e_target = max(e_target, rows[-1][4])
    return 100.0 * e_target / n_target


def model_en_sat(states, i, t_target, n_target):
    rows = [r for r in _trophy_points(states, i) if r[4] is not None]
    if len(rows) < 3 or not n_target:
        return model_en_loglog(states, i, t_target, n_target)
    best = None
    for tau in SAT_TAU_H:
        for p in SAT_P:
            gs = [weibull_g(r[0], tau, p) for r in rows]
            denom = sum(g * g for g in gs)
            if denom < 1e-12:
                continue
            c = sum(g * r[4] for g, r in zip(gs, rows)) / denom
            if c <= 0:
                continue
            sse = sum((r[4] - c * g) ** 2 for g, r in zip(gs, rows))
            rel = sse / max(sum(r[4] ** 2 for r in rows), 1e-9)
            if best is None or rel < best[0]:
                best = (rel, c, tau, p)
    if best is None:
        return model_en_loglog(states, i, t_target, n_target)
    _, c, tau, p = best
    e_target = c * weibull_g(t_target, tau, p)
    e_target = max(e_target, rows[-1][4])
    return 100.0 * e_target / n_target


MODELS = {
    "persist": model_persist,
    "logit_ln": model_logit_ln,
    "logit_ln24": model_logit_ln24,
    "logit_sat": model_logit_sat,
    "en_loglog": model_en_loglog,
    "en_power": model_en_power,
    "en_sat": model_en_sat,
}


# --------------------------------------------------------------------------
# backtest + ensemble
# --------------------------------------------------------------------------

def _n_forecast_for(states, t_target):
    last_n = None
    for st in reversed(states):
        if st.get("N"):
            last_n = st["N"]
            break
    if last_n is None:
        return None
    n = forecast_count(states, "N", t_target, last_n)
    return clamp(n, last_n, last_n * 400.0)


def backtest(states, n_tr, skip):
    """Fit each model on truncated histories, score held-out last states.

    Returns (global_err, per_trophy_err, bt_gap_days, ensemble_bt_errors)
    where errors are mean |logit error| of the median; ensemble_bt_errors
    is the list of per-trophy squared logit errors of an equal-weight
    ensemble, used for the spread scale. skip: trophy indices to exclude
    from pooling (the always-100% trophy).
    """
    cuts = []
    if len(states) >= 4:
        cuts.append((states[:-1], states[-1], 0.50))
    if len(states) >= 5:
        cuts.append((states[:-2], states[-2], 0.20))
    if len(states) >= 6:
        # longer-horizon cut: one-step backtests favour persistence;
        # predicting two states out rewards models that extrapolate
        cuts.append((states[:-3], states[-1], 0.30))
    if not cuts:
        return None, None, None, None
    g_err = {m: 0.0 for m in MODEL_NAMES}
    g_w = {m: 0.0 for m in MODEL_NAMES}
    t_err = [{m: 0.0 for m in MODEL_NAMES} for _ in range(n_tr)]
    t_w = [{m: 0.0 for m in MODEL_NAMES} for _ in range(n_tr)]
    ens_sq = []
    gap_days = 0.0
    for sub, held, cw in cuts:
        n_held = _n_forecast_for(sub, held["t"])
        gap_days = max(gap_days, (held["t"] - sub[-1]["t"]) / 24.0)
        for i in range(n_tr):
            y = held["rates"][i]
            if y is None:
                continue
            ly = logit(y / 100.0)
            preds = {}
            for m in MODEL_NAMES:
                try:
                    preds[m] = MODELS[m](sub, i, held["t"], n_held)
                except Exception:
                    preds[m] = y
            ens = sum(logit(clamp(p, 0.01, 99.99) / 100.0) for p in preds.values()) / len(preds)
            if i not in skip:
                ens_sq.append((ens - ly) ** 2)
            for m, p in preds.items():
                e = abs(logit(clamp(p, 0.01, 99.99) / 100.0) - ly)
                t_err[i][m] += cw * e
                t_w[i][m] += cw
                if i not in skip:
                    g_err[m] += cw * e
                    g_w[m] += cw
    glob = {m: (g_err[m] / g_w[m] if g_w[m] > 0 else 1.0) for m in MODEL_NAMES}
    per = []
    for i in range(n_tr):
        per.append({m: (t_err[i][m] / t_w[i][m] if t_w[i][m] > 0 else glob[m])
                    for m in MODEL_NAMES})
    return glob, per, max(gap_days, 0.05), ens_sq


def ensemble_weights(glob, per_i):
    w = {}
    for m in MODEL_NAMES:
        w[m] = math.exp(-glob[m] / 0.12) * math.exp(-(per_i[m] - glob[m]) / 0.30)
    tot = sum(w.values()) or 1.0
    w = {m: v / tot for m, v in w.items()}
    # cap any single model at 0.65, redistribute the excess
    for _ in range(3):
        over = [m for m in w if w[m] > 0.65]
        if not over:
            break
        excess = sum(w[m] - 0.65 for m in over)
        for m in over:
            w[m] = 0.65
        rest = [m for m in w if m not in over]
        rtot = sum(w[m] for m in rest) or 1.0
        for m in rest:
            w[m] += excess * (w[m] / rtot)
    return w


# --------------------------------------------------------------------------
# constraints: story chain monotonicity, platinum cap
# --------------------------------------------------------------------------

def _pava_nonincreasing(vals):
    """Pool-adjacent-violators: closest non-increasing sequence (L2)."""
    blocks = [[v, 1] for v in vals]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] < blocks[i + 1][0]:
            tot = blocks[i][0] * blocks[i][1] + blocks[i + 1][0] * blocks[i + 1][1]
            cnt = blocks[i][1] + blocks[i + 1][1]
            blocks[i] = [tot / cnt, cnt]
            del blocks[i + 1]
            if i > 0:
                i -= 1
        else:
            i += 1
    out = []
    for v, c in blocks:
        out.extend([v] * c)
    return out


def apply_constraints(trophies, medians):
    med = list(medians)
    # story chain: trophies whose description is a chapter completion,
    # in id order, must be non-increasing (sequential unlocks)
    chain = []
    for i, tr in enumerate(trophies):
        desc = (tr.get("description") or "").lower()
        if ("complete chapter" in desc or "complete gone fishing" in desc
                or "complete the final mission" == desc.strip().rstrip(".")):
            chain.append(i)
    chain.sort(key=lambda i: trophies[i]["id"])
    if len(chain) >= 3:
        fixed = _pava_nonincreasing([med[i] for i in chain])
        for i, v in zip(chain, fixed):
            med[i] = v
    # platinum needs every other trophy: its rate caps at the minimum
    plat = [i for i, tr in enumerate(trophies) if tr.get("tier") == "platinum"]
    if plat and len(med) > 1:
        others = [med[i] for i in range(len(med)) if i not in plat]
        cap = min(others) if others else med[plat[0]]
        for i in plat:
            med[i] = min(med[i], cap)
    return med


# --------------------------------------------------------------------------
# flat-history fallback (practice cases: one frozen launch-night page)
# --------------------------------------------------------------------------

def fallback_median(states, i, tier, t_target, n_target, p_last):
    """Dilution prior when there is nothing to fit: players keep growing
    from when the frozen count was fresh (first state's time), earners
    grow at a tier-prior log-log slope."""
    rows = _trophy_points(states, i)
    if not rows or not n_target:
        return 0.01
    r_last = rows[-1][3]
    e_last = max(rows[-1][4] or 0.0, 0.05)
    t_fresh = states[0]["t"]
    x_fresh = x_of(t_fresh)
    slope_e = TIER_PRIOR.get(tier, 0.15)
    e_target = e_last * math.exp(slope_e * (x_of(t_target) - x_fresh))
    med = 100.0 * e_target / n_target
    return clamp(med, 0.01, 99.99)


# --------------------------------------------------------------------------
# case forecast
# --------------------------------------------------------------------------

def forecast_case(case):
    trophies = case["trophies"]
    n_tr = len(trophies)
    try:
        target = parse_time(case["target"])
    except Exception:
        target = parse_time(case["at"])
    t_target = hours(target)
    horizon_days = max(float(case.get("horizon_minutes") or 0.0), 0.0) / 1440.0

    states = build_states(case)
    if not states:
        return {"trophies": [{"id": tr["id"], "q": [0.01, 0.05, 0.1, 0.5, 2.0]}
                              for tr in trophies]}
    # thin a very long state list so fits stay fast (Theil-Sen is O(n^2));
    # keep an even spread plus the last state
    if len(states) > 60:
        step = len(states) / 60.0
        thinned = [states[int(i * step)] for i in range(60)]
        if thinned[-1] is not states[-1]:
            thinned.append(states[-1])
        states = thinned

    # always-100% trophies (the first-mission trophy): keep at 100
    always_full = []
    for i in range(n_tr):
        rs = [st["rates"][i] for st in states if st["rates"][i] is not None]
        always_full.append(bool(rs) and all(v >= 99.995 for v in rs))
    skip = {i for i in range(n_tr) if always_full[i]}

    p_last = states[-1]["N"]
    n_target = _n_forecast_for(states, t_target) or p_last
    log_growth = math.log(max(n_target / p_last, 1.0))

    glob, per, gap_days, ens_sq = backtest(states, n_tr, skip)
    have_bt = glob is not None
    if have_bt:
        bt_scale = math.sqrt(sum(ens_sq) / len(ens_sq)) if ens_sq else 0.4
        horizon_factor = clamp((horizon_days / gap_days) ** 0.5, 1.0, 6.0)
        sigma_bt = bt_scale * horizon_factor
    else:
        sigma_bt = None

    medians = [0.0] * n_tr
    sigmas = [0.0] * n_tr
    for i, tr in enumerate(trophies):
        if always_full[i]:
            medians[i] = 100.0
            sigmas[i] = 0.0
            continue
        tier = tr.get("tier") or "bronze"
        if len(states) >= 3 and have_bt:
            w = ensemble_weights(glob, per[i])
            preds = {}
            for m in MODEL_NAMES:
                try:
                    preds[m] = MODELS[m](states, i, t_target, n_target)
                except Exception:
                    preds[m] = model_persist(states, i, t_target, n_target)
            lp = {m: logit(clamp(p, 0.01, 99.99) / 100.0) for m, p in preds.items()}
            med_logit = sum(w[m] * lp[m] for m in MODEL_NAMES)
            medians[i] = 100.0 * logistic(med_logit)
            disagree = math.sqrt(sum(w[m] * (lp[m] - med_logit) ** 2 for m in MODEL_NAMES))
            e_last = 0.0
            for st in reversed(states):
                if st["E"][i] is not None:
                    e_last = st["E"][i]
                    break
            sigma_pois = min(0.6 / math.sqrt(max(e_last, 1.0)), 0.6)
            sigma = math.sqrt(sigma_bt ** 2 + (0.8 * disagree) ** 2
                              + sigma_pois ** 2 + 0.22 ** 2) * SIGMA_SCALE
            sigmas[i] = clamp(sigma, 0.35, 1.7)
        elif len(states) >= 3:
            # enough states to fit but not to backtest: plain ensemble mean
            preds = []
            for m in MODEL_NAMES:
                try:
                    preds.append(MODELS[m](states, i, t_target, n_target))
                except Exception:
                    pass
            if preds:
                medians[i] = 100.0 * logistic(
                    sum(logit(clamp(p, 0.01, 99.99) / 100.0) for p in preds) / len(preds))
            else:
                medians[i] = model_persist(states, i, t_target, n_target)
            sigmas[i] = clamp(math.sqrt(0.8 ** 2 + (0.5 * math.sqrt(horizon_days)) ** 2
                                        + (0.35 * log_growth) ** 2), 0.5, 2.2)
        else:
            medians[i] = fallback_median(states, i, tier, t_target, n_target, p_last)
            sigmas[i] = clamp(math.sqrt(0.8 ** 2 + (0.5 * math.sqrt(horizon_days)) ** 2
                                        + (0.35 * log_growth) ** 2), 0.5, 2.2)
        medians[i] = clamp(medians[i], 0.01, 99.99)

    medians = apply_constraints(trophies, medians)

    rows = []
    for i, tr in enumerate(trophies):
        if always_full[i]:
            rows.append({"id": tr["id"], "q": [99.0, 99.8, 100, 100, 100]})
            continue
        base = logit(medians[i] / 100.0)
        qs = []
        prev = -1.0
        for z in Z:
            q = round4(100.0 * logistic(base + z * sigmas[i]))
            q = clamp(float(q), 0.01, 99.99)
            if q < prev:
                q = prev
            qs.append(q)
            prev = q
        rows.append({"id": tr["id"], "q": qs})
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
        if case.get("levels") not in (None, LEVELS):
            raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
        out = forecast_case(case)
        stem = name[:-3] if name.endswith(".in") else name
        with open(os.path.join(out_dir, stem + ".out"), "w", encoding="utf-8") as f:
            f.write(json.dumps(out, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
