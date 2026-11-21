#!/usr/bin/env python3
"""Forecast GTA VI PlayStation 5 trophy earn rates on Exophase.

Entry for make.fun gta6-trophy-rarity-forecast-01. The program reads every
``<case>.in`` from the directory named by ``MAKEFUN_INPUT_DIR`` (default
``/input``) and writes a matching ``<case>.out`` to ``MAKEFUN_OUTPUT_DIR``
(default ``/output``): for each of the game's trophies, the 5%, 25%, 50%,
75% and 95% quantiles of its earn rate (percent of tracked players holding
it) at the case's target.

The data is Exophase's trophy page as the house recorded it: a sequence of
reads, each with the players tracked and every trophy's earn rate. Two facts
about that page drive the forecast:

* The page refreshes on Exophase's own schedule, in steps. Between refreshes
  it prints the same numbers for hours, even on launch night while players
  pour in. When it refreshes, the tracked-player count jumps (launch night:
  638 to 3,868) and almost every rate falls, because the new players have
  earned far less than the early cohort. A flat history therefore does not
  mean nothing is happening; it means a refresh is due.
* Once the page is refreshing regularly, each trophy's rate moves on its own
  track: the opening beats (the first chapter, the first activity) climb
  fast as the player base plays through them, while the grind and
  completion trophies stay near zero for days. Those tracks are close to
  straight lines of logit(rate) against log time since release, so the
  forecast fits one per trophy and reads it at the target.

Model
-----
First the history is reduced to its distinct reads: consecutive reads that
print the same page carry no new information. Then:

* Flat history (one distinct read, or none). The page has not refreshed in
  the case's window, so the forecast assumes the next refresh lands before
  the target and re-bases the player count the way the first launch-night
  refresh did: tracked players multiply by about six (638 to 3,868, as the
  competition rules state). Earned counts are carried over unchanged -- the
  new players have had only hours, and in their first hours they earn
  almost nothing except the opening beats -- except the trophies the page
  already shows at 100%, which stay at 100%, and the opening beats
  themselves (the first chapter, the fishing the game opens with), for
  which about one new player in fifty is assumed to have caught up. The
  spread is a half logit unit, honest about a refresh the case has not
  seen. On the practice cases this branch scores about 0.24 against the
  house baseline's 1.51 (local scorer, refresh answers estimated from the
  rules' own figures).
* Moving history (several distinct reads). For each trophy, a least-squares
  line of logit(rate) against ln(1 + hours since release) is fitted on the
  reads of the last 36 hours after the largest jump in the tracked-player
  count (the whole history when fewer than four distinct reads fall in
  that window), so the fit follows the current regime rather than the
  launch-night page. The median is the line read at the target, anchored
  on the trophy's rate in the last read, with the slope clamped to a sane
  range. The spread combines the horizon (a random-walk half logit per
  root day, as the house baseline uses), the fit's own residual, and a
  step term for the next refresh. Trophies the page shows at 100% stay at
  100%; trophies at 0.00% stay near 0.01%, with the spread reaching up
  for the small chance a first earner appears.

No pooling across trophies: on the launch-night to 20 November backtest
(rates at 21 Nov 00:27 UTC, read from Exophase's public page), pooling
slopes by tier scored far worse than per-trophy slopes, because the
trophies split into fast risers and near-flat diluters and the tier does
not separate them. The per-trophy fit scored 0.12 on that backtest
against persistence's 0.06 over eleven hours -- the trend only pays over
the final's four-and-a-half-day horizon, which is why the spread, not the
slope, is damped for short horizons.

Standard library only. No network. Deterministic: the same case always
writes the same bytes.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# First launch-night refresh, as the competition rules state it: tracked
# players 638 -> 3,868. The flat-history branch assumes the next refresh
# re-bases the player count by about the same multiple.
REFRESH_MULTIPLE = 3868.0 / 638.0
# Share of the new players assumed to have caught up with an opening beat
# (the first chapter, the fishing the game opens with) by the first refresh:
# about one in fifty, in their first hours of play.
OPENING_CATCH_UP = 0.02
# Spread of the flat-history forecast, in logit units: the refresh is a
# step the case has not seen, so the band is wide on purpose.
FLAT_SIGMA = 0.5
# Trend branch: random-walk spread per root day, fit-residual floor and cap,
# and a step term for the next refresh, all in logit units.
TREND_SIGMA_PER_ROOT_DAY = 0.45
RESID_FLOOR = 0.15
RESID_CAP = 1.0
REFRESH_STEP = 0.25
# Slope of logit(rate) per unit of ln(1 + hours), clamped to this range.
SLOPE_CLAMP = 3.0
# The trend fit uses reads from this many hours before the checkpoint.
TREND_WINDOW_HOURS = 36.0
MIN_WINDOW_READS = 4


def parse_time(text):
    """An ISO 8601 UTC timestamp to an aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def clamp_rate(pct):
    """An earn rate into the range the logit is defined on."""
    return min(99.99, max(0.01, pct))


def logit(pct):
    return math.log(clamp_rate(pct) / (100.0 - clamp_rate(pct)))


def inv_logit(lg):
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    """Round to 4 decimals, half up, as the house starter rounds it."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def quantile_row(median_pct, sigma):
    """Five quantiles around a median rate, in percent, non-decreasing."""
    if median_pct >= 100.0:
        return [100, 100, 100, 100, 100]
    centre = logit(median_pct)
    row = []
    for z in Z:
        value = round4(min(100.0, max(0.01, inv_logit(centre + z * sigma))))
        row.append(value)
    for k in range(1, len(row)):
        if row[k] < row[k - 1]:
            row[k] = row[k - 1]
    return row


def distinct_reads(history):
    """History reduced to reads that print a different page, oldest first.

    Consecutive reads with the same players-tracked count and the same
    rates are one read: the page had not refreshed between them. The time
    kept is the first read's.
    """
    seen = []
    last_key = None
    for read in history:
        key = (read.get("players_tracked"), tuple(read.get("rates") or ()))
        if key != last_key:
            seen.append(read)
            last_key = key
    return seen


def hours_since_release(moment, release_midnight):
    """Hours from the release midnight (UTC) to a moment; never below -0.9."""
    hours = (moment - release_midnight).total_seconds() / 3600.0
    return max(hours, -0.9)


def last_known_rates(case):
    """Each trophy's rate in the last read that lists it, oldest fallback."""
    rates = [None] * len(case["trophies"])
    for read in case["history"]:
        for i, value in enumerate(read.get("rates") or ()):
            if value is not None:
                rates[i] = value
    return rates


def is_opening_beat(trophy):
    """True for the beats every new player reaches in their first hours.

    The game opens with fishing and its first chapter; a new player who
    has played for an hour or two has done those, and little else. The
    description is the only signal a flat history carries, so the match is
    deliberately narrow: the first chapter, and fishing.
    """
    text = (trophy.get("description") or "").lower()
    return "chapter 1" in text or "fishing" in text


def flat_forecast(case, reads, release_midnight):
    """Forecast from a history that has not moved: a refresh is due.

    The next refresh re-bases the player count by the first refresh's
    multiple, earned counts carry over, and only the opening beats gain
    new earners among the new players.
    """
    last = reads[-1]
    players_now = last.get("players_tracked") or 0
    players_new = max(players_now * REFRESH_MULTIPLE, players_now + 1)
    newcomers = players_new - players_now
    horizon_days = case["horizon_minutes"] / 1440.0
    sigma = FLAT_SIGMA * math.sqrt(1.0 + horizon_days)
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rate = last["rates"][i] if last.get("rates") else None
        if rate is None:
            rate = 0.0
        if rate >= 100.0:
            rows.append({"id": trophy["id"], "q": [100, 100, 100, 100, 100]})
            continue
        earned = rate * players_now / 100.0
        if is_opening_beat(trophy):
            earned += OPENING_CATCH_UP * newcomers
        median = 100.0 * earned / players_new
        rows.append({"id": trophy["id"], "q": quantile_row(median, sigma)})
    return {"trophies": rows}


def fit_line(xs, ys):
    """Least-squares line through (x, y) points; returns (intercept, slope).

    With fewer than two points there is no slope to fit.
    """
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    if n == 1:
        return ys[0], 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 0.0:
        return mean_y, 0.0
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    return mean_y - slope * mean_x, slope


def trend_forecast(case, reads, release_midnight):
    """Forecast from a history that moves: one logit-log-time line a trophy.

    The line is fitted on the reads of the last 36 hours (the whole history
    when fewer than four distinct reads fall in that window) and read at the
    target, anchored on the trophy's rate in the last read.
    """
    checkpoint = parse_time(case["at"])
    target = parse_time(case["target"])
    horizon_days = case["horizon_minutes"] / 1440.0
    x_target = math.log1p(hours_since_release(target, release_midnight))

    # The fit follows the current regime. A refresh that multiplies the
    # tracked-player count re-bases every rate, so reads from before the
    # largest such jump in the history are left out when enough reads
    # remain after it: they belong to a smaller, earlier cohort, not to
    # the player base whose rates the target is about.
    regime_start = 0
    biggest_jump = 1.0
    jump_after = 0
    for k in range(1, len(reads)):
        before = reads[k - 1].get("players_tracked") or 0
        after = reads[k].get("players_tracked") or 0
        if before > 0 and after / before > biggest_jump:
            biggest_jump = after / before
            jump_after = k
    if biggest_jump > 2.0 and len(reads) - jump_after >= MIN_WINDOW_READS:
        regime_start = jump_after

    window_start = checkpoint.timestamp() - TREND_WINDOW_HOURS * 3600.0
    window = [r for r in reads[regime_start:]
              if parse_time(r["at"]).timestamp() >= window_start]
    if len(window) < MIN_WINDOW_READS:
        window = reads[regime_start:]
    if len(window) < MIN_WINDOW_READS:
        window = reads

    last = reads[-1]
    last_rates = last_known_rates(case)
    base_sigma = TREND_SIGMA_PER_ROOT_DAY * math.sqrt(horizon_days)
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        last_rate = last_rates[i]
        if last_rate is None:
            last_rate = 0.0
        if last_rate >= 100.0:
            rows.append({"id": trophy["id"], "q": [100, 100, 100, 100, 100]})
            continue
        xs = []
        ys = []
        for read in window:
            rates = read.get("rates") or ()
            if i >= len(rates) or rates[i] is None:
                continue
            moment = parse_time(read["at"])
            xs.append(math.log1p(hours_since_release(moment, release_midnight)))
            ys.append(logit(rates[i]))
        if len(xs) >= 2:
            intercept, raw_slope = fit_line(xs, ys)
            if len(xs) > 2:
                resid = math.sqrt(
                    sum((y - (intercept + raw_slope * x)) ** 2
                        for x, y in zip(xs, ys)) / len(xs)
                )
            else:
                resid = 0.0
            slope = max(-SLOPE_CLAMP, min(SLOPE_CLAMP, raw_slope))
        else:
            slope = 0.0
            resid = 0.0
        resid = min(RESID_CAP, max(RESID_FLOOR, resid))
        sigma = math.sqrt(base_sigma ** 2 + resid ** 2 + REFRESH_STEP ** 2)
        x_last = math.log1p(
            hours_since_release(parse_time(last["at"]), release_midnight)
        )
        median_logit = logit(last_rate) + slope * (x_target - x_last)
        median = min(99.99, max(0.01, inv_logit(median_logit)))
        rows.append({"id": trophy["id"], "q": quantile_row(median, sigma)})
    return {"trophies": rows}


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case.get("history"):
        raise ValueError(f"{name} has no history")
    release = parse_time(case["game"]["release_date"] + "T00:00:00Z")
    reads = distinct_reads(case["history"])
    if len(reads) <= 1:
        return flat_forecast(case, reads, release)
    return trend_forecast(case, reads, release)


def main():
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    names = sorted(name for name in os.listdir(INPUT_DIR) if name.endswith(".in"))
    for name in names:
        case = json.loads((INPUT_DIR / name).read_text(encoding="utf-8"))
        forecast = forecast_case(case, name)
        out = json.dumps(forecast, separators=(",", ":")) + "\n"
        if len(out.encode("utf-8")) > 65536:
            raise ValueError(f"{name}: forecast over the 65,536 byte cap")
        (OUTPUT_DIR / (name[: -len(".in")] + ".out")).write_text(out, encoding="utf-8")


if __name__ == "__main__":
    main()
