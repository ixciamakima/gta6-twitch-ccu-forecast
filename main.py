#!/usr/bin/env python3
"""Twitch CCU forecaster for big game launches (make.fun: gta6-twitch-ccu-forecast-01).

Reads one case file per run from $MAKEFUN_INPUT_DIR (else /input) and writes
<hours 2..24 quantile forecasts> to $MAKEFUN_OUTPUT_DIR (else /output).

Model summary
-------------
A launch day on Twitch is two waves. Wave 1 (APAC + Europe) unlocks at
midnight local time region by region and is already over by the forecast
checkpoint: it lives in the hours before the UTC day starts. Wave 2 (the
Americas) is what the forecast day is mostly about. The model therefore
anchors every curve on W, the peak hourly audience of wave 1, instead of the
hour-1 print alone, and learns how past launches converted their wave 1 into
a release-day curve.

For each past launch ("comparable") shipped inside the case file we compute
features that are all observable at the checkpoint: ln W, ln(hour1 / W),
ln(viewers per channel at the wave-1 peak) and ln(viewers at the last
pre-midnight hour / W), plus the day curve expressed as log-ratios to W.
The forecast is a geometric-mean ensemble of four members:

  1. per-hour kernel-weighted ridge regression of the log-ratios on all
     four features, with the weighted residual quantiles shrunk toward the
     conditional median (spread 0.6);
  2. the same ridge on a three-feature subset;
  3. a nearest-neighbour member: the ten comparables closest to the target
     in standardised feature space contribute their empirical ratio
     quantiles, recentred on member 1's median;
  4. a kernel-weighted empirical-quantile pool over all comparables,
     weighted by |ln W - ln W_target|.

The ensemble's log-quantile curves are lightly smoothed across neighbouring
hours. Everything is deterministic: no randomness, no network, standard
library only. If anything about a case is unusable the program falls back
to a persistence forecast with a widening spread, so it always emits a
valid forecast file.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
HOURS = tuple(range(2, 25))
FEATS4 = ("lnW", "lnvpcW", "lnh1", "last1")
FEATS3 = ("lnW", "lnvpcW", "last1")
Z = (-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722)


def parse_ts(text):
    """Parse an ISO-8601 timestamp into an aware datetime in UTC."""
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def bucket_hours(points, start, key):
    """Group points into whole-hour buckets relative to `start`.

    Returns {hour_offset: (mean_viewers, last_channels)} where offset 0 is
    the hour starting at `start`. Points without a usable viewer count are
    skipped.
    """
    acc = {}
    for pt in points:
        viewers = pt.get("viewers")
        raw_ts = pt.get(key)
        if viewers is None or raw_ts is None:
            continue
        try:
            t = parse_ts(raw_ts)
        except (ValueError, TypeError):
            continue
        offset = int((t - start).total_seconds() // 3600)
        slot = acc.setdefault(offset, [0.0, 0, None])
        slot[0] += float(viewers)
        slot[1] += 1
        if pt.get("channels") is not None:
            slot[2] = pt.get("channels")
    return {off: (tot / n, ch) for off, (tot, n, ch) in acc.items() if n > 0}


def extract_features(hourly_points, start, base, ts_key, curve_points=None):
    """Checkpoint-observable features for a target or a comparable.

    hourly_points: the series used for the pre-day window (history_hourly
    for a target, hourly for a comparable). curve_points: for comparables,
    the same series, used to build the release-day curve. Returns None when
    the record is too thin to learn from.
    """
    buckets = bucket_hours(hourly_points, start, ts_key)
    pre = {off: buckets[off] for off in range(-16, 0) if off in buckets and buckets[off][0] > 0}
    if len(pre) < 6 or not base or base <= 0:
        return None
    w_off = max(pre, key=lambda o: pre[o][0])
    W, ch_w = pre[w_off]
    if W <= 0:
        return None
    last_off = max(pre)
    hour1 = buckets.get(0)
    ch1 = hour1[1] if hour1 else None
    feats = {
        "base": float(base),
        "W": float(W),
        "lnW": math.log(W),
        "lnh1": math.log(base / W),
        "lnvpcW": math.log(W / ch_w) if ch_w else 0.0,
        "last1": math.log(pre[last_off][0] / W),
    }
    if curve_points is not None:
        day = bucket_hours(curve_points, start, ts_key)
        ratios = {}
        for h in HOURS:
            row = day.get(h - 1)
            if row and row[0] and row[0] > 0:
                ratios[h] = math.log(row[0] / W)
        if len(ratios) < 12:
            return None
        feats["ratiosW"] = ratios
    return feats


def wquantile(pairs, tau):
    """Weighted quantile of [(value, weight)] with linear interpolation."""
    pairs = sorted(pairs)
    tot = sum(w for _, w in pairs)
    if tot <= 0:
        return pairs[len(pairs) // 2][0]
    cum = 0.0
    prev_v, prev_c = pairs[0][0], 0.0
    for v, w in pairs:
        cum += w
        if cum / tot >= tau:
            if cum == w:
                return v
            frac = (tau * tot - prev_c) / w
            return prev_v + frac * (v - prev_v)
        prev_v, prev_c = v, cum
    return pairs[-1][0]


def solve_linear(mat, rhs):
    """Solve a small dense linear system by Gauss-Jordan elimination."""
    d = len(rhs)
    aug = [row[:] + [rhs[i]] for i, row in enumerate(mat)]
    for col in range(d):
        piv = max(range(col, d), key=lambda r: abs(aug[r][col]))
        if abs(aug[piv][col]) < 1e-12:
            continue
        aug[col], aug[piv] = aug[piv], aug[col]
        for r in range(d):
            if r != col and abs(aug[col][col]) > 1e-12:
                fac = aug[r][col] / aug[col][col]
                for cc in range(col, d + 1):
                    aug[r][cc] -= fac * aug[col][cc]
    return [aug[i][d] / aug[i][i] if abs(aug[i][i]) > 1e-12 else 0.0 for i in range(d)]


def ridge_forecast(tf, pool, feats, spread, ridge=1.0, bw=0.35):
    """Per-hour kernel-weighted ridge on log-ratios to W + residual quantiles."""
    out = {}
    for h in HOURS:
        pts = [(f, f["ratiosW"][h]) for f in pool if h in f.get("ratiosW", {})]
        if not pts:
            out[h] = [float(tf["base"])] * 5
            continue
        ws = [math.exp(-bw * abs(f["lnW"] - tf["lnW"])) for f, _ in pts]
        X = [[1.0] + [f[k] - tf[k] for k in feats] for f, _ in pts]
        yv = [y for _, y in pts]
        d = len(X[0])
        mat = [[sum(ws[i] * X[i][a] * X[i][b] for i in range(len(pts)))
                + (ridge if a == b and a > 0 else 0.0)
                for b in range(d)] for a in range(d)]
        rhs = [sum(ws[i] * X[i][a] * yv[i] for i in range(len(pts))) for a in range(d)]
        beta = solve_linear(mat, rhs)
        resid = sorted(
            (yv[i] - sum(beta[j] * X[i][j] for j in range(d)), ws[i])
            for i in range(len(pts))
        )
        med_r = wquantile(resid, 0.5)
        qs = [beta[0] + med_r + (wquantile(resid, t) - med_r) * spread for t in LEVELS]
        out[h] = [tf["W"] * math.exp(q) for q in qs]
    return out


def hybrid_forecast(tf, pool, k=10, spread=0.6):
    """Nearest-neighbour ratio quantiles recentred on the ridge median."""
    base_fc = ridge_forecast(tf, pool, FEATS4, spread=1.0)
    std = {}
    for key in FEATS4:
        vals = [f[key] for f in pool]
        mu = sum(vals) / len(vals)
        std[key] = math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals)) or 1.0

    def dist(f):
        return math.sqrt(sum(((tf[key] - f[key]) / std[key]) ** 2 for key in FEATS4))

    neighbours = sorted(pool, key=dist)[:k]
    out = {}
    for h in HOURS:
        pairs = [(f["ratiosW"][h], 1.0) for f in neighbours if h in f.get("ratiosW", {})]
        if not pairs:
            out[h] = base_fc[h]
            continue
        med_log = math.log(base_fc[h][2] / tf["W"])
        nn_med = wquantile(pairs, 0.5)
        out[h] = [tf["W"] * math.exp(med_log + (wquantile(pairs, t) - nn_med) * spread)
                  for t in LEVELS]
    return out


def wpool_forecast(tf, pool, spread=0.6):
    """Kernel-weighted empirical quantiles of the log-ratios to W."""
    pools = {h: [] for h in HOURS}
    for f in pool:
        w = math.exp(-abs(math.log(f["W"]) - math.log(tf["W"])))
        for h, val in f["ratiosW"].items():
            pools[h].append((val, w))
    out = {}
    for h in HOURS:
        pairs = pools[h]
        if not pairs:
            out[h] = [float(tf["base"])] * 5
            continue
        qs = [wquantile(pairs, t) for t in LEVELS]
        med = qs[2]
        qs = [med + (q - med) * spread for q in qs]
        out[h] = [tf["W"] * math.exp(q) for q in qs]
    return out


def ensemble_forecast(tf, pool):
    """Geometric mean of the four members + light smoothing across hours."""
    members = [
        ridge_forecast(tf, pool, FEATS4, spread=0.6),
        ridge_forecast(tf, pool, FEATS3, spread=0.6),
        hybrid_forecast(tf, pool, k=10, spread=0.6),
        wpool_forecast(tf, pool, spread=0.6),
    ]
    out = {}
    for h in HOURS:
        out[h] = [math.exp(sum(math.log(m[h][i]) for m in members) / len(members))
                  for i in range(5)]
    smoothed = {}
    for h in HOURS:
        acc = [0.0] * 5
        wsum = 0.0
        for dh, wgt in ((-1, 0.25), (0, 1.0), (1, 0.25)):
            if h + dh in out:
                for i in range(5):
                    acc[i] += math.log(out[h + dh][i]) * wgt
                wsum += wgt
        smoothed[h] = [math.exp(a / wsum) for a in acc]
    return smoothed


def fallback_forecast(base):
    """Persistence with a widening normal spread (the house baseline's shape)."""
    p = max(float(base), 1.0)
    out = {}
    for h in HOURS:
        s = 0.15 * math.sqrt(h - 1)
        out[h] = [p * math.exp(z * s) for z in Z]
    return out


def target_features(case, base):
    """Features for the case being forecast, from its own recorded history."""
    start = parse_ts(case["day_start"])
    for series, key in ((case.get("history_hourly"), "from"), (case.get("history"), "at")):
        if series:
            feats = extract_features(series, start, base, key)
            if feats is not None:
                return feats
    return None


def pool_features(case):
    """Features for every usable comparable shipped inside the case."""
    pool = []
    for comp in case.get("comparables") or []:
        try:
            start = parse_ts(comp["day_start"])
            hourly = comp.get("hourly") or []
            day = bucket_hours(hourly, start, "from")
            hour1 = day.get(0)
            base = hour1[0] if hour1 else None
            feats = extract_features(hourly, start, base, "from", curve_points=hourly)
            if feats is not None:
                pool.append(feats)
        except Exception:
            continue
    return pool


def forecast_case(case):
    """Forecast one case: {hour: [q05, q25, q50, q75, q95]} for hours 2..24."""
    known = case.get("known") or []
    base = 1.0
    for row in known:
        if row.get("hour") == 1 and row.get("viewers"):
            base = float(row["viewers"])
            break
    else:
        if known and known[0].get("viewers"):
            base = float(known[0]["viewers"])
    base = max(base, 1.0)
    try:
        tf = target_features(case, base)
        pool = pool_features(case)
        if tf is None or len(pool) < 5:
            return fallback_forecast(base)
        raw = ensemble_forecast(tf, pool)
    except Exception:
        return fallback_forecast(base)
    out = {}
    for h in HOURS:
        qs = []
        prev = 0.0
        for q in raw[h]:
            q = float(q)
            if not math.isfinite(q):
                q = prev if prev > 0 else base
            q = min(max(q, 1.0), 1e9)
            q = max(q, prev)
            qs.append(round(q, 2))
            prev = qs[-1]
        out[h] = qs
    return out


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    for name in names:
        out_name = name[:-3] + ".out"
        try:
            with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
                case = json.load(fh)
            fc = forecast_case(case)
        except Exception:
            # A malformed case file must not sink the run: emit a validly
            # shaped fallback forecast for it and move on.
            fc = fallback_forecast(1.0)
        payload = {"hours": [{"hour": h, "q": [round(float(q), 2) for q in fc[h]]}
                             for h in HOURS]}
        with open(os.path.join(out_dir, out_name), "w", encoding="utf-8") as fh:
            json.dump(payload, fh, separators=(",", ":"))
            fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
