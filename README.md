# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/0xDimzy/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/0xDimzy/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Backtest-weighted ensemble forecaster for GTA VI PS5 trophy earn rates on
Exophase, one week after launch. It reads a case (every recorded read of the
trophy page before the checkpoint) and writes five quantiles per trophy for
the earn rate at the target instant, scored by pinball loss on the logit
scale.

## Method

Exophase refreshes a game's counts in steps, so the read history collapses
to a handful of distinct states. An earn rate is earners / players tracked:
earners only grow, and the denominator grows as new players sync, which
mostly dilutes rates while maturing cohorts push late trophies up. The
program fits seven small models per trophy at run time:

- persistence (last rate),
- logit-linear in ln(1 + hours) over all states and over the last ~30 hours,
- logit-saturating `A - B * exp(-x / tau)` on a tau grid,
- earners/players decomposition three ways: log-log (Theil-Sen), a power
  law of earners in players (the dilution exponent), and saturating
  Weibull-CDF earners; players themselves are forecast by a blend of a
  log-log line and a saturating curve.

Weights come from an internal backtest: every model is refit on the states
before the last one (and before the second-to-last, and two states out) and
scored on the held-out state, pooled across trophies with a per-trophy
adjustment shrunk toward the pooled weights. The spread on the logit scale
combines the ensemble's backtest residual scale grown to the forecast
horizon, the disagreement between model medians, and a Poisson term for
trophies with few earners. Story-chapter medians are made non-increasing
along the chapter chain (unlocks are sequential) and the platinum median is
capped by every other trophy's. With too few distinct states to backtest —
the practice cases are one frozen launch-night page — it falls back to
dilution priors: players keep growing from when the frozen count was fresh
and earners grow at tier-prior rates.

Deterministic, Python 3.13 standard library only, no network.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

In the grader's container the program is run as `python3 /bot/main.py`,
reading `/input/` and writing `/output/`.

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
