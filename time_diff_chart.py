#!/usr/bin/env python3
"""Compute consecutive timestamp deltas in milliseconds and plot diagnostics."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import statistics
from pathlib import Path
from typing import Sequence

try:  # pragma: no cover - optional dependency
    import matplotlib.pyplot as plt  # type: ignore
except ImportError:  # pragma: no cover - environment dependent
    plt = None

try:  # pragma: no cover - optional dependency
    from statsmodels.tsa.stattools import adfuller  # type: ignore
except ImportError:  # pragma: no cover - environment dependent
    adfuller = None

DEFAULT_MAX_AUTO_BINS = 200
DEFAULT_MIN_AUTO_BINS = 10
DEFAULT_ROLL_WINDOW = 1000


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Path to the CSV file containing Date and Time columns",
    )
    parser.add_argument(
        "--date-column",
        default="Date",
        help="Column name holding the date (default: Date)",
    )
    parser.add_argument(
        "--time-column",
        default="Time",
        help="Column name holding the time (default: Time)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on rows processed (after header).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="If provided, save the plot to this file instead of showing it.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=None,
        help="Number of bins for the histogram (default: adaptive)",
    )
    parser.add_argument(
        "--show-series",
        action="store_true",
        help="Add a subplot showing the delta series over sample index.",
    )
    parser.add_argument(
        "--show-rolling",
        action="store_true",
        help="Add a subplot showing rolling mean/std of the delta series.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=DEFAULT_ROLL_WINDOW,
        help=f"Window size for rolling statistics (default: {DEFAULT_ROLL_WINDOW}).",
    )
    parser.add_argument(
        "--full-range",
        action="store_true",
        help="Display the full range on the histogram x-axis (default trims extreme outliers).",
    )
    parser.add_argument(
        "--adf-test",
        action="store_true",
        help="Run Augmented Dickey-Fuller stationarity test on the delta series.",
    )
    parser.add_argument(
        "--value-min",
        type=float,
        help="Minimum value (ms) for histogram x-axis and y-axes on delta plots.",
    )
    parser.add_argument(
        "--value-max",
        type=float,
        help="Maximum value (ms) for histogram x-axis and y-axes on delta plots.",
    )
    parser.add_argument(
        "--value-tick",
        type=float,
        help="Major tick spacing (ms) for delta axes (histogram x-axis, series/rolling y-axis).",
    )
    return parser.parse_args(argv)


def read_timestamps(
    csv_path: Path,
    date_column: str,
    time_column: str,
    limit: int | None,
) -> list[dt.datetime]:
    timestamps: list[dt.datetime] = []
    with csv_path.open(newline="", encoding="utf-8-sig") as fp:
        reader = csv.DictReader(fp)
        for idx, row in enumerate(reader):
            if limit is not None and idx >= limit:
                break
            date_val = (row.get(date_column) or "").strip()
            time_val = (row.get(time_column) or "").strip()
            if not date_val or not time_val:
                continue
            stamp = to_datetime(date_val, time_val)
            if stamp is not None:
                timestamps.append(stamp)
    return timestamps


def to_datetime(date_str: str, time_str: str) -> dt.datetime | None:
    combined = f"{date_str} {time_str}"
    patterns = (
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y/%m/%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
    )
    for pattern in patterns:
        try:
            return dt.datetime.strptime(combined, pattern)
        except ValueError:
            continue
    return None


def consecutive_deltas(timestamps: Sequence[dt.datetime]) -> list[float]:
    deltas_ms: list[float] = []
    for prev, curr in zip(timestamps, timestamps[1:]):
        diff_ms = (curr - prev).total_seconds() * 1000.0
        deltas_ms.append(diff_ms)
    return deltas_ms


def describe(values: Sequence[float], sorted_values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {}
    stats = {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "stdev": statistics.pstdev(values),
    }
    stats["p005"] = percentile(sorted_values, 0.5)
    stats["p995"] = percentile(sorted_values, 99.5)
    return stats


def percentile(sorted_values: Sequence[float], pct: float) -> float:
    if not sorted_values:
        return math.nan
    if pct <= 0:
        return sorted_values[0]
    if pct >= 100:
        return sorted_values[-1]
    k = (len(sorted_values) - 1) * (pct / 100.0)
    lower_idx = math.floor(k)
    upper_idx = math.ceil(k)
    lower = sorted_values[lower_idx]
    upper = sorted_values[upper_idx]
    if lower_idx == upper_idx:
        return lower
    return lower + (upper - lower) * (k - lower_idx)


def choose_bins(count: int) -> int:
    adaptive = max(DEFAULT_MIN_AUTO_BINS, count // 2000)
    return min(DEFAULT_MAX_AUTO_BINS, adaptive)


def rolling_mean_std(values: Sequence[float], window: int) -> tuple[list[float], list[float]]:
    if window <= 1:
        raise ValueError("Rolling window must be > 1")
    means: list[float] = []
    stds: list[float] = []
    acc = 0.0
    acc_sq = 0.0
    buffer: list[float] = []

    for val in values:
        buffer.append(val)
        acc += val
        acc_sq += val * val
        if len(buffer) > window:
            old = buffer.pop(0)
            acc -= old
            acc_sq -= old * old
        if len(buffer) == window:
            mean = acc / window
            variance = max(acc_sq / window - mean * mean, 0.0)
            std = math.sqrt(variance)
            means.append(mean)
            stds.append(std)
    return means, stds


def apply_value_axis(ax, axis: str, vmin: float | None, vmax: float | None, tick: float | None) -> None:
    if axis == "x":
        if vmin is not None or vmax is not None:
            ax.set_xlim(left=vmin, right=vmax)
        if tick:
            start = vmin if vmin is not None else ax.get_xlim()[0]
            end = vmax if vmax is not None else ax.get_xlim()[1]
            ticks = list(_frange(start, end, tick))
            ax.set_xticks(ticks)
    else:
        if vmin is not None or vmax is not None:
            ax.set_ylim(bottom=vmin, top=vmax)
        if tick:
            start = vmin if vmin is not None else ax.get_ylim()[0]
            end = vmax if vmax is not None else ax.get_ylim()[1]
            ticks = list(_frange(start, end, tick))
            ax.set_yticks(ticks)


def _frange(start: float, stop: float, step: float) -> Sequence[float]:
    if step <= 0 or start > stop:
        return []
    values = []
    current = start
    safety = 0
    while current <= stop + 1e-9 and safety < 2000:
        values.append(current)
        current += step
        safety += 1
    return values


def plot_histogram(
    deltas: Sequence[float],
    sorted_deltas: Sequence[float],
    stats: dict[str, float],
    bins_arg: int | None,
    output: Path | None,
    show_series: bool,
    show_rolling: bool,
    rolling_stats: tuple[list[float], list[float]] | None,
    full_range: bool,
    value_min: float | None,
    value_max: float | None,
    value_tick: float | None,
) -> None:
    if plt is None:
        raise RuntimeError(
            "matplotlib is required for plotting. Install it with 'pip install matplotlib'."
        )

    bins = bins_arg or choose_bins(len(deltas))
    if bins < 1:
        bins = DEFAULT_MIN_AUTO_BINS

    ncols = 1 + int(show_series) + int(show_rolling)
    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=(6 + 4 * (ncols - 1), 6),
        squeeze=False,
    )
    col_idx = 0
    ax_hist = axes[0][col_idx]
    col_idx += 1

    ax_hist.hist(deltas, bins=bins, color="#2a6fbb", edgecolor="#08306b", alpha=0.85)
    ax_hist.set_title("Consecutive Timestamp Differences")
    ax_hist.set_xlabel("Delta (milliseconds)")
    ax_hist.set_ylabel("Frequency")
    ax_hist.grid(axis="y", alpha=0.3)

    if stats:
        text_lines = [
            f"count: {int(stats['count'])}",
            f"min: {stats['min']:.3f} ms",
            f"p0.5: {stats['p005']:.3f} ms",
            f"median: {stats['median']:.3f} ms",
            f"p99.5: {stats['p995']:.3f} ms",
            f"max: {stats['max']:.3f} ms",
            f"mean: {stats['mean']:.3f} ms",
            f"stdev: {stats['stdev']:.3f} ms",
        ]
        ax_hist.text(
            0.98,
            0.98,
            "\n".join(text_lines),
            transform=ax_hist.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"facecolor": "#f5f9ff", "alpha": 0.85, "boxstyle": "round"},
        )

    if not full_range and stats and value_min is None and value_max is None:
        lo = stats.get("p005")
        hi = stats.get("p995")
        if lo is not None and hi is not None and math.isfinite(lo) and math.isfinite(hi) and lo < hi:
            ax_hist.set_xlim(lo, hi)
            ax_hist.axvline(stats["median"], color="#ff7f0e", linestyle="--", linewidth=1.2, label="median")
            ax_hist.legend(loc="upper left")

    apply_value_axis(ax_hist, "x", value_min, value_max, value_tick)

    if show_series:
        ax_series = axes[0][col_idx]
        col_idx += 1
        ax_series.plot(deltas, color="#d62728", linewidth=0.6)
        ax_series.set_title("Delta Series")
        ax_series.set_xlabel("Sample index")
        ax_series.set_ylabel("Delta (milliseconds)")
        ax_series.grid(alpha=0.3)
        if not full_range and stats and value_min is None and value_max is None:
            lo = stats.get("p005")
            hi = stats.get("p995")
            if lo is not None and hi is not None and math.isfinite(lo) and math.isfinite(hi):
                ax_series.set_ylim(lo, hi)
        apply_value_axis(ax_series, "y", value_min, value_max, value_tick)

    if show_rolling and rolling_stats is not None:
        means, stds = rolling_stats
        ax_roll = axes[0][col_idx]
        x_vals = range(len(deltas) - len(means), len(deltas))
        ax_roll.plot(x_vals, means, color="#1b9e77", linewidth=0.9, label="rolling mean")
        ax_roll.plot(x_vals, stds, color="#7570b3", linewidth=0.9, label="rolling std")
        ax_roll.set_title("Rolling Mean & Std")
        ax_roll.set_xlabel("Sample index")
        ax_roll.set_ylabel("Milliseconds")
        ax_roll.grid(alpha=0.3)
        ax_roll.legend()
        apply_value_axis(ax_roll, "y", value_min, value_max, value_tick)

    fig.tight_layout()

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output)
        print(f"Histogram saved to {output}")
    else:
        plt.show()


def run_adf_test(deltas: Sequence[float]) -> None:
    if adfuller is None:
        print("statsmodels not available; install it with 'pip install statsmodels' to run ADF test.")
        return

    print("Running Augmented Dickey-Fuller test...")
    result = adfuller(deltas, autolag="AIC")
    stat, pvalue, used_lag, used_obs = result[:4]
    crit_vals = result[4]

    print(f"  Test statistic: {stat:.6f}")
    print(f"  p-value       : {pvalue:.6g}")
    print(f"  Used lags     : {used_lag}")
    print(f"  Observations  : {used_obs}")
    print("  Critical values:")
    for level, value in crit_vals.items():
        print(f"    {level}%: {value:.6f}")
    if pvalue < 0.05:
        print("  => Reject null hypothesis (series is likely stationary).")
    else:
        print("  => Fail to reject null (series may not be stationary).")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.show_rolling and args.rolling_window <= 1:
        print("Rolling window must be greater than 1.")
        return 1

    timestamps = read_timestamps(args.csv_path, args.date_column, args.time_column, args.limit)
    if len(timestamps) < 2:
        print("Not enough timestamps to compute deltas.")
        return 1

    deltas = consecutive_deltas(timestamps)
    if not deltas:
        print("No valid deltas computed.")
        return 1

    sorted_deltas = sorted(deltas)
    stats = describe(deltas, sorted_deltas)

    print("Inter-sample delta statistics (milliseconds):")
    for key in ("count", "min", "p005", "median", "p995", "max", "mean", "stdev"):
        val = stats.get(key)
        if val is None or not math.isfinite(val):
            continue
        if key == "count":
            print(f"  {key:>6}: {int(val)}")
        else:
            print(f"  {key:>6}: {val:.3f}")

    rolling_stats: tuple[list[float], list[float]] | None = None
    if args.show_rolling:
        means, stds = rolling_mean_std(deltas, args.rolling_window)
        if not means:
            print("Rolling window larger than series; skipping rolling stats plot.")
        else:
            rolling_stats = (means, stds)
            print(
                "Rolling mean (last window): "
                f"min={min(means):.3f}, max={max(means):.3f}, last={means[-1]:.3f}"
            )
            print(
                "Rolling std (last window): "
                f"min={min(stds):.3f}, max={max(stds):.3f}, last={stds[-1]:.3f}"
            )

    if args.adf_test:
        run_adf_test(deltas)

    try:
        plot_histogram(
            deltas,
            sorted_deltas,
            stats,
            args.bins,
            args.output,
            args.show_series,
            args.show_rolling,
            rolling_stats,
            args.full_range,
            args.value_min,
            args.value_max,
            args.value_tick,
        )
    except RuntimeError as err:
        print(err)
        print("Plotting skipped because matplotlib is not available.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
