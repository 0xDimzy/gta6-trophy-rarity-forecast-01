# GTA VI Trophy Rarity Forecaster

![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)
[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01/0xDimzy/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01/0xDimzy/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts every GTA VI PlayStation 5 trophy's earn rate on Exophase at the
case target — five quantiles (5%, 25%, 50%, 75%, 95%) for each of the 55
trophies.

## Method

`main.py` (Python 3.13, standard library only, no network) fits its model at
run time from the case file itself — nothing is hard-coded to any launch:

1. Exophase refreshes a game's counts on its own schedule, not continuously:
   between refreshes the page is frozen, and at a refresh the
   players-tracked count jumps while almost every rate falls, because the
   new players have not earned the trophies yet. A forecast that says
   "rates stay put" misses every refresh, so this program models the two
   quantities a rate is made of — earners and players — separately.
2. Players: log(players_tracked) against ln(1 + hours since launch) with a
   Theil-Sen slope over the history's distinct refresh points, shrunk
   halfway toward a slope of 1 (players grow roughly linearly with time
   since launch while a game is new). When the history is completely flat
   — no refresh yet, as on launch night — the slope prior of 1 is used,
   anchored at the first read, because the frozen count is stale since the
   page last changed.
3. Earners per trophy: E = rate × players / 100 at each refresh point,
   log(E) fitted the same way, per trophy, with a Theil-Sen slope shrunk
   halfway toward the median slope of its tier. When the history is flat
   the slope prior is tier-based (bronze 0.15, silver 0.10, gold 0.05,
   platinum 0). Median rate = 100 × E_target / P_target.
4. Spread: quantiles are logistic(logit(median) + z_tau × sigma) with
   sigma combining a floor for refresh risk, a random-walk term growing
   with the horizon, a dilution-uncertainty term growing with how far the
   player count has to jump, and the trophy's own fit residual. The band is
   deliberately wide — a refresh can shift every rate at once.

Output is exactly one row per trophy, five non-decreasing quantiles,
deterministic: two runs write identical bytes.

Run: `python3 main.py`, reading `$MAKEFUN_INPUT_DIR` (else `/input/`) and
writing `$MAKEFUN_OUTPUT_DIR` (else `/output/`), one `.out` per `.in` file.
