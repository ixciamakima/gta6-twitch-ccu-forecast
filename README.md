# gta6-twitch-ccu-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/ixciamakima/gta6-twitch-ccu-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/ixciamakima/gta6-twitch-ccu-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

A forecaster for the hour-by-hour Twitch audience (CCU) of a big game launch,
built for make.fun's GTA VI Twitch CCU forecast. Given everything recorded up
to the end of a launch day's first hour, it predicts five quantiles
(5/25/50/75/95%) of the viewer count for each of hours 2–24 UTC.

## How it works

A rolling-midnight launch day is two waves. Wave 1 (APAC + Europe) unlocks
region by region before the UTC day even starts, so by the checkpoint it is
already history: the hours before midnight hold a complete first release
wave. Wave 2 (the Americas) is what the forecast day is mostly about. The
program therefore anchors every curve on **W**, the peak hourly audience of
wave 1, instead of on the hour-1 print alone, and learns from the past
launches shipped inside each case file how a wave 1 converts into a
release-day curve.

For every comparable launch it computes checkpoint-observable features —
log W, log(hour-1 / W), log viewers-per-channel at the wave-1 peak, and the
wave-1 tail level at the last pre-midnight hour — and expresses the launch's
day curve as log-ratios to its own W. The forecast is a deterministic
geometric-mean ensemble of four members:

1. per-hour kernel-weighted ridge regression of the log-ratios on all four
   features, with weighted residual quantiles shrunk toward the conditional
   median;
2. the same ridge on a three-feature subset;
3. a nearest-neighbour member (ten closest comparables in standardised
   feature space) whose empirical ratio quantiles are recentred on the
   ridge median;
4. a kernel-weighted empirical-quantile pool over all comparables.

The ensemble's log-quantile curves are lightly smoothed across neighbouring
hours, clamped to a valid range, and forced non-decreasing. If a case is too
thin to learn from, the program falls back to a persistence forecast with a
widening spread, so it always emits a valid forecast. Standard library only,
no network, byte-for-byte deterministic across runs.

On the kit's four practice launches it scores a mean pinball loss of about
0.156 (typical miss ≈ 28%), against the house baseline's ≈ 0.387
(typical miss ≈ 83%).

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<dir with .in case files> MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Each `<id>.in` in the input directory produces one `<id>.out` in the output
directory: `{"hours":[{"hour":2,"q":[q05,q25,q50,q75,q95]}, …]}` for hours
2–24. With no environment variables set it reads `/input` and writes
`/output`, as the grader's sandbox expects.

## License

MIT — see [LICENSE](LICENSE).

---

<sub>Entered on [make.fun](https://make.fun/?ref=z00wg8vw&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
