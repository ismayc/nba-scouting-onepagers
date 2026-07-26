# Opponent scouting one-pagers, auto-generated

**What.** One page per team for the full 2025-26 regular season,
tomorrow's opponent as a coaching staff would consume it: keys up top, then
the shot profile vs league, personnel decomposed into diet vs making, and
the most-used lineups. All 30 reports live in `reports/` as paste-ready
markdown and print-ready self-contained HTML.

**How it stays honest.** The generator invents no new statistics; it
composes the repo's gated machinery:

- Shot pricing is the shot-quality model's committed 2023-24 coefficients
  scored on 2025-26, a transfer the season-out backtest validated
  (worst out-of-season <abbr title="Split shots into ten bins by predicted make probability and compare predicted vs actual rates per bin; the reported number is the worst bin's gap.">decile</abbr> gap 0.014).
- Player <abbr title="Actual minus expected points per 100 shots: conversion above or below what the shot diet itself explains. Positive means making shots the model prices as hard.">shot-making</abbr> is **<abbr title="Estimate the spread of true skill across the league from the data, then pull each individual's noisy estimate toward the league mean in proportion to its noise.">empirical-Bayes</abbr> shrunk** (the rule from
  `shot-quality-study/python/08_projection.py`, imported rather than
  reimplemented) before any bullet is allowed to call someone a
  <abbr title="Actual minus expected points per 100 shots: conversion above or below what the shot diet itself explains. Positive means making shots the model prices as hard.">shot-maker</abbr>. Small-sample hot streaks don't survive the <abbr title="The fraction of an observed number that survives empirical-Bayes regression: 1 means fully trusted, 0 means replaced by the league mean.">shrinkage</abbr>.
- Lineup net ratings come from the per-team Base+Advanced files; player
  <abbr title="Regularized adjusted plus-minus: a ridge regression crediting each player with net points per 100 possessions while adjusting for the other nine players on the floor.">RAPM</abbr> from the opponent-adjusted 2025-26 lineup model, <abbr title="Confidence interval.">CI</abbr> caveat stated
  on the page.
- The "Keys" bullets are rule-generated from thresholds, so every claim on
  the page traces to a number in a table below it.

**Gate.** `run_all_checks.sh` regenerates the PHX report in memory and
fails if the committed copy differs byte-for-byte. The reports cannot
silently go stale against the data or drift by hand-editing. Pure-function
tests live in `tests/python/test_scouting.py`.

<!-- terms -->
> **Terms used in this analysis.** Dotted-underlined terms anywhere below repeat these definitions on hover ([full glossary](https://github.com/ismayc/basketball-data-science/blob/main/docs/glossary.md)).
>
> - **shot-making**: Actual minus expected points per 100 shots: conversion above or below what the shot diet itself explains. Positive means making shots the model prices as hard.
> - **RAPM**: Regularized adjusted plus-minus: a ridge regression crediting each player with net points per 100 possessions while adjusting for the other nine players on the floor.
> - **decile calibration**: Split shots into ten bins by predicted make probability and compare predicted vs actual rates per bin; the reported number is the worst bin's gap.
> - **empirical Bayes**: Estimate the spread of true skill across the league from the data, then pull each individual's noisy estimate toward the league mean in proportion to its noise.
> - **shrinkage weight**: The fraction of an observed number that survives empirical-Bayes regression: 1 means fully trusted, 0 means replaced by the league mean.
> - **CI**: Confidence interval.
<!-- /terms -->

## Run

```
python scouting/build_onepager.py PHX          # one team
python scouting/build_onepager.py LAL BOS      # several
python scouting/build_onepager.py --all        # all 30
python scouting/build_onepager.py --check      # staleness gate
```
