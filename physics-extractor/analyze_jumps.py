#!/usr/bin/env python3
"""
Analyze Bonk.io jump recordings produced by physics-extractor spy scripts.

Usage examples:
    python analyze_jumps.py data/0-jumps-up-tap.json
    python analyze_jumps.py data/0-tap-up

The script prints summary statistics, optional plots, and aggregate tables
inside physics-extractor/analysis/.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class JumpStats:
    index: int
    start_ms: float
    end_ms: float
    duration_s: float
    apex_time_s: float
    apex_y: float
    apex_height: float
    takeoff_velocity: float
    gravity: float
    ball_px: Optional[float]
    apex_height_units: Optional[float]
    takeoff_velocity_units: Optional[float]
    gravity_units: Optional[float]


@dataclass
class RunStats:
    index: int
    start_ms: float
    end_ms: float
    duration_s: float
    direction: str
    acceleration_px: float
    acceleration_units: Optional[float]
    terminal_velocity_px: float
    terminal_velocity_units: Optional[float]


@dataclass
class TapStats:
    index: int
    start_ms: float
    duration_s: float
    press_duration_s: float
    direction: str
    thrust_acceleration_px: Optional[float]
    thrust_acceleration_units: Optional[float]
    release_acceleration_px: Optional[float]
    release_acceleration_units: Optional[float]
    peak_speed_px: Optional[float]
    peak_speed_units: Optional[float]
    glide_duration_s: Optional[float]
    distance_px: Optional[float]
    distance_units: Optional[float]


def load_json_recording(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw = json.loads(path.read_text())
    t = np.array([float(row[0]) for row in raw])
    x = np.array([float(row[1]) for row in raw])
    y = np.array([float(row[2]) for row in raw])
    width = np.full_like(y, np.nan, dtype=float)
    return t, x, y, width


def load_csv_recording(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    times: List[float] = []
    xs: List[float] = []
    ys: List[float] = []
    widths: List[float] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row["Time_ms"]))
            xs.append(float(row["X_px"]))
            ys.append(float(row["Y_px"]))
            bw = row.get("BallWidth_px")
            bh = row.get("BallHeight_px")
            if bw is not None and bh is not None:
                widths.append((float(bw) + float(bh)) / 2.0)
            elif bw is not None:
                widths.append(float(bw))
            else:
                widths.append(float("nan"))
    t = np.array(times, dtype=float)
    x = np.array(xs, dtype=float)
    y = np.array(ys, dtype=float)
    width = np.array(widths, dtype=float)
    return t, x, y, width


def load_recording(path: Path, trim_ms: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if path.suffix.lower() == ".json":
        t, x, y, width = load_json_recording(path)
    elif path.suffix.lower() == ".csv":
        t, x, y, width = load_csv_recording(path)
    else:
        raise ValueError(f"Unsupported file format: {path}")

    # Drop duplicate timestamps by keeping the last sample at each time.
    dedup_mask = np.concatenate((np.diff(t) != 0, [True]))
    t = t[dedup_mask]
    x = x[dedup_mask]
    y = y[dedup_mask]
    width = width[dedup_mask]
    if trim_ms > 0 and t.size > 0:
        cutoff = t[-1] - trim_ms
        mask = t <= cutoff
        if not np.any(mask):
            mask = np.ones_like(t, dtype=bool)
        t = t[mask]
        x = x[mask]
        y = y[mask]
        width = width[mask]
    return t, x, y, width


def detect_airborne_segments(
    t: np.ndarray, y: np.ndarray, baseline: float, threshold: float = 0.5
) -> List[Tuple[int, int]]:
    airborne = y < baseline - threshold
    segments: List[Tuple[int, int]] = []
    start = None
    for idx, is_airborne in enumerate(airborne):
        if is_airborne and start is None:
            start = idx
        if not is_airborne and start is not None:
            segments.append((start, idx - 1))
            start = None
    if start is not None:
        segments.append((start, len(y) - 1))
    return segments


def fit_parabola(ts: np.ndarray, ys: np.ndarray) -> Tuple[float, float, float]:
    # Fit y = a t^2 + b t + c
    coeffs = np.polyfit(ts, ys, 2)
    return coeffs[0], coeffs[1], coeffs[2]


def analyze_jumps(
    t: np.ndarray, y: np.ndarray, width: Optional[np.ndarray] = None
) -> Tuple[List[JumpStats], float]:
    baseline = float(np.max(y))
    segments = detect_airborne_segments(t, y, baseline)
    stats: List[JumpStats] = []
    width = width if width is not None else np.full_like(y, np.nan, dtype=float)
    for idx, (start, end) in enumerate(segments):
        seg_slice = slice(start, end + 1)
        seg_t = (t[seg_slice] - t[start]) / 1000.0  # seconds relative to takeoff
        seg_y = y[seg_slice]
        seg_w = width[seg_slice]

        if len(seg_t) < 5:
            continue
        a, b, c = fit_parabola(seg_t, seg_y)
        gravity = 2 * a  # because y = y0 + v0*t + 0.5*g*t^2
        apex_time = float(-b / (2 * a))
        apex_y = float(np.polyval([a, b, c], apex_time))
        apex_height = baseline - apex_y
        finite_w = seg_w[~np.isnan(seg_w)]
        ball_px = float(np.mean(finite_w)) if finite_w.size else None
        if ball_px and ball_px != 0:
            apex_units = apex_height / ball_px
            takeoff_units = b / ball_px
            gravity_units = gravity / ball_px
        else:
            apex_units = None
            takeoff_units = None
            gravity_units = None

        stats.append(
            JumpStats(
                index=idx,
                start_ms=float(t[start]),
                end_ms=float(t[end]),
                duration_s=float((t[end] - t[start]) / 1000.0),
                apex_time_s=apex_time,
                apex_y=apex_y,
                apex_height=apex_height,
                takeoff_velocity=b,
                gravity=gravity,
                ball_px=ball_px,
                apex_height_units=apex_units,
                takeoff_velocity_units=takeoff_units,
                gravity_units=gravity_units,
            )
        )
    mean_gravity = float(np.mean([s.gravity for s in stats]))
    return stats, mean_gravity


def detect_horizontal_segments(
    t: np.ndarray,
    x: np.ndarray,
    velocity_threshold: float = 50.0,
    min_duration: float = 0.4,
    max_duration: Optional[float] = 5.0,
) -> List[Tuple[int, int]]:
    if len(t) < 3:
        return []
    vx = np.gradient(x, t) * 1000.0  # px/s
    moving = np.abs(vx) > velocity_threshold
    segments: List[Tuple[int, int]] = []
    start = None
    for idx, is_moving in enumerate(moving):
        if is_moving and start is None:
            start = idx
        if not is_moving and start is not None:
            end = idx - 1
            if end > start:
                duration = (t[end] - t[start]) / 1000.0
                if duration >= min_duration and (max_duration is None or duration <= max_duration):
                    segments.append((start, end))
            start = None
    if start is not None:
        end = len(t) - 1
        duration = (t[end] - t[start]) / 1000.0
        if duration >= min_duration and (max_duration is None or duration <= max_duration):
            segments.append((start, end))
    return segments


def analyze_horizontal(
    t: np.ndarray,
    x: np.ndarray,
    width: np.ndarray,
    velocity_threshold: float = 50.0,
    min_duration: float = 0.4,
    max_duration: Optional[float] = 5.0,
) -> List[RunStats]:
    segments = detect_horizontal_segments(
        t, x, velocity_threshold=velocity_threshold, min_duration=min_duration, max_duration=max_duration
    )
    stats: List[RunStats] = []
    for idx, (start, end) in enumerate(segments):
        seg_slice = slice(start, end + 1)
        seg_t = (t[seg_slice] - t[start]) / 1000.0
        seg_x = x[seg_slice]
        seg_w = width[seg_slice]
        if len(seg_t) < 5:
            continue
        a, b, c = np.polyfit(seg_t, seg_x, 2)
        acceleration = 2 * a  # px/s^2
        duration = seg_t[-1]
        terminal_velocity = 2 * a * duration + b
        mean_ball = float(np.mean(seg_w[~np.isnan(seg_w)])) if np.any(~np.isnan(seg_w)) else None
        acc_units = acceleration / mean_ball if mean_ball else None
        vel_units = terminal_velocity / mean_ball if mean_ball else None
        direction = "right" if acceleration > 0 else "left"
        stats.append(
            RunStats(
                index=idx,
                start_ms=float(t[start]),
                end_ms=float(t[end]),
                duration_s=float(duration),
                direction=direction,
                acceleration_px=float(acceleration),
                acceleration_units=acc_units,
                terminal_velocity_px=float(terminal_velocity),
                terminal_velocity_units=vel_units,
            )
        )
    return stats


def smooth_series(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def analyze_horizontal_tap(
    t: np.ndarray,
    x: np.ndarray,
    width: np.ndarray,
    speed_start: float = 5.0,
    speed_stop: float = 0.5,
    press_window: float = 0.2,
    smooth_window: int = 5,
) -> List[TapStats]:
    if len(t) < 5:
        return []
    t_sec = (t - t[0]) / 1000.0
    vx = np.gradient(x, t_sec, edge_order=1)
    vx = smooth_series(vx, smooth_window)
    moving = np.abs(vx) > speed_start

    idx = np.argmax(moving) if np.any(moving) else None
    if idx is None or not moving[idx]:
        return []
    pre_samples = 2
    start = max(0, idx - pre_samples)
    end = len(vx) - 1

    stats: List[TapStats] = []
    ball_px = float(np.nanmean(width)) if np.any(~np.isnan(width)) else None

    for seg_idx, (start, end) in enumerate([(start, end)]):
        if end - start < 5:
            continue
        peak_relative = np.argmax(np.abs(vx[start : end + 1]))
        peak = start + peak_relative
        direction = "right" if vx[peak] >= 0 else "left"

        thrust_end_time = min(t_sec[peak], t_sec[start] + press_window)
        thrust_end_idx = np.searchsorted(t_sec, thrust_end_time)
        if thrust_end_idx <= start + 1:
            thrust_end_idx = min(start + 3, end)
        press_duration = max(t_sec[thrust_end_idx] - t_sec[start], 1e-6)
        v_start = float(abs(vx[start]))

        release_start = max(thrust_end_idx, start + 1)
        release_slice = slice(release_start, end + 1)
        release_t = t_sec[release_slice]
        release_v = vx[release_slice]
        release_accel = None
        if len(release_t) >= 2 and release_t[-1] - release_t[0] > 1e-6:
            coeffs = np.polyfit(release_t, release_v, 1)
            release_accel = float(coeffs[0])

        peak_speed = float(np.max(np.abs(vx[start : end + 1])))
        delta_v = max(peak_speed - v_start, 0.0)
        thrust_accel_px = float(delta_v / press_duration) if press_duration > 0 else None
        thrust_accel_units = (
            (thrust_accel_px / ball_px) if (thrust_accel_px is not None and ball_px) else None
        )
        glide_duration = t_sec[end] - t_sec[peak]
        total_duration = t_sec[end] - t_sec[start]
        press_duration = t_sec[thrust_end_idx] - t_sec[start]
        distance = float(np.abs(x[end] - x[start]))

        stats.append(
            TapStats(
                index=seg_idx,
                start_ms=float(t[start]),
                duration_s=float(total_duration),
                press_duration_s=float(press_duration),
                direction=direction,
                thrust_acceleration_px=thrust_accel_px,
                thrust_acceleration_units=thrust_accel_units,
                release_acceleration_px=release_accel,
                release_acceleration_units=(release_accel / ball_px) if (release_accel is not None and ball_px) else None,
                peak_speed_px=peak_speed,
                peak_speed_units=(peak_speed / ball_px) if (ball_px and peak_speed is not None) else None,
                glide_duration_s=float(glide_duration) if glide_duration is not None else None,
                distance_px=distance,
                distance_units=(distance / ball_px) if (ball_px and distance is not None) else None,
            )
        )
    return stats


def plot_overview(
    t: np.ndarray,
    y: np.ndarray,
    stats: Sequence[JumpStats],
    baseline: float,
    out_dir: Path,
    dataset_name: str,
) -> None:
    t_sec = (t - t[0]) / 1000.0
    plt.figure(figsize=(12, 4))
    plt.plot(t_sec, y, label="y(t)")
    plt.axhline(baseline, color="gray", linestyle="--", label="rest height")
    for jump in stats:
        start = (jump.start_ms - t[0]) / 1000.0
        end = (jump.end_ms - t[0]) / 1000.0
        plt.axvspan(start, end, color="orange", alpha=0.1)
    plt.gca().invert_yaxis()
    plt.xlabel("time (s)")
    plt.ylabel("vertical position (px)")
    plt.title(f"{dataset_name}: all jumps")
    plt.legend()
    out_path = out_dir / f"{dataset_name}-overview.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_normalized_jumps(
    t: np.ndarray,
    y: np.ndarray,
    stats: Sequence[JumpStats],
    baseline: float,
    out_dir: Path,
    dataset_name: str,
) -> None:
    plt.figure(figsize=(6, 4))
    for jump in stats:
        mask = (t >= jump.start_ms) & (t <= jump.end_ms)
        seg_t = (t[mask] - jump.start_ms) / 1000.0
        seg_height = baseline - y[mask]
        plt.plot(seg_t, seg_height, alpha=0.7)
    plt.xlabel("time since takeoff (s)")
    plt.ylabel("height above floor (px)")
    plt.title(f"{dataset_name}: normalized jump profiles")
    out_path = out_dir / f"{dataset_name}-normalized.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_apexes(stats: Sequence[JumpStats], out_dir: Path, dataset_name: str) -> None:
    apexes = [s.apex_height for s in stats]
    plt.figure(figsize=(6, 4))
    plt.bar(range(len(apexes)), apexes)
    plt.xlabel("jump #")
    plt.ylabel("apex height (px)")
    plt.title(f"{dataset_name}: apex comparison")
    out_path = out_dir / f"{dataset_name}-apexes.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def safe_mean(values: Iterable[Optional[float]]) -> Optional[float]:
    data = [v for v in values if v is not None]
    if not data:
        return None
    return float(np.mean(data))


def mean_abs(values: Iterable[Optional[float]]) -> Optional[float]:
    data = [abs(v) for v in values if v is not None]
    if not data:
        return None
    return float(np.mean(data))


def resolve_inputs(paths: Sequence[Path]) -> List[Path]:
    files: List[Path] = []
    for path in paths:
        if path.is_dir():
            for suffix in ("*.csv", "*.json"):
                files.extend(sorted(path.glob(suffix)))
        elif path.is_file():
            files.append(path)
    return sorted(files)


def format_val(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="recording file or directory (supports multiple)",
    )
    parser.add_argument("--skip-plots", action="store_true", help="disable PNG output (faster)")
    parser.add_argument(
        "--mode",
        choices=("vertical", "horizontal"),
        default="vertical",
        help="analyze jumps (vertical) or horizontal runs",
    )
    parser.add_argument(
        "--horizontal-style",
        choices=("hold", "tap"),
        default="hold",
        help="horizontal analysis type: constant hold or single tap/release",
    )
    parser.add_argument(
        "--trim-ms",
        type=float,
        default=0.0,
        help="discard the last N milliseconds of each recording (useful when recordings stop late)",
    )
    parser.add_argument(
        "--h-velocity-threshold",
        type=float,
        default=50.0,
        help="horizontal mode: minimum speed (px/s) to treat as moving",
    )
    parser.add_argument(
        "--h-min-duration",
        type=float,
        default=0.4,
        help="horizontal mode: minimum segment duration in seconds",
    )
    parser.add_argument(
        "--h-max-duration",
        type=float,
        default=5.0,
        help="horizontal mode: maximum segment duration in seconds (set <=0 to disable)",
    )
    parser.add_argument(
        "--tap-speed-start",
        type=float,
        default=3.0,
        help="tap mode: threshold speed (px/s) to detect movement start",
    )
    parser.add_argument(
        "--tap-speed-stop",
        type=float,
        default=0.5,
        help="tap mode: threshold speed (px/s) to detect movement end",
    )
    parser.add_argument(
        "--tap-press-window",
        type=float,
        default=0.2,
        help="tap mode: seconds of data to use for thrust acceleration fit",
    )
    parser.add_argument(
        "--tap-smooth-window",
        type=int,
        default=3,
        help="tap mode: moving-average window (samples) for velocity smoothing",
    )
    args = parser.parse_args()

    inputs = resolve_inputs(args.paths)
    if not inputs:
        raise SystemExit("No recordings found.")

    out_dir = Path("physics-extractor") / "analysis"
    if not args.skip_plots:
        out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for path in inputs:
        t, x, y, width = load_recording(path, trim_ms=args.trim_ms)
        if len(t) == 0:
            print(f"Skipping {path} (empty recording)")
            continue

        dataset_name = path.stem

        if args.mode == "vertical":
            stats, mean_gravity = analyze_jumps(t, y, width)
            if len(stats) == 0:
                print(f"Skipping {path} (no jumps detected)")
                continue

            baseline = float(np.max(y))

            if not args.skip_plots:
                plot_overview(t, y, stats, baseline, out_dir, dataset_name)
                plot_normalized_jumps(t, y, stats, baseline, out_dir, dataset_name)
                plot_apexes(stats, out_dir, dataset_name)

            print(f"\nAnalysis for {dataset_name}")
            print(f"Detected jumps: {len(stats)}")
            print(f"Mean gravity (px/s^2): {mean_gravity:.2f}")
            print(f"Rest height y: {baseline:.2f}")
            ball_px = safe_mean([s.ball_px for s in stats])
            if ball_px:
                print(f"Mean on-screen ball diameter: {ball_px:.2f} px")
            print()
            header = "jump duration(s) apex_y apex_height takeoff_v(px/s) gravity(px/s^2) apex_t(s)"
            print(header)
            for s in stats:
                print(
                    f"{s.index:02d} {s.duration_s:8.3f} {s.apex_y:7.2f} {s.apex_height:11.2f} "
                    f"{s.takeoff_velocity:14.2f} {s.gravity:15.2f} {s.apex_time_s:9.3f}"
                )

            summaries.append(
                {
                    "name": dataset_name,
                    "count": len(stats),
                    "ball_px": ball_px,
                    "apex_px": safe_mean([s.apex_height for s in stats]),
                    "apex_ball": safe_mean([s.apex_height_units for s in stats]),
                    "gravity_px": mean_gravity,
                    "gravity_ball": safe_mean([s.gravity_units for s in stats]),
                    "takeoff_px": safe_mean([s.takeoff_velocity for s in stats]),
                    "takeoff_ball": safe_mean([s.takeoff_velocity_units for s in stats]),
                }
            )
        else:
            if args.horizontal_style == "hold":
                stats = analyze_horizontal(
                    t,
                    x,
                    width,
                    velocity_threshold=args.h_velocity_threshold,
                    min_duration=args.h_min_duration,
                    max_duration=None if args.h_max_duration and args.h_max_duration <= 0 else args.h_max_duration,
                )
            else:
                stats = analyze_horizontal_tap(
                    t,
                    x,
                    width,
                    speed_start=args.tap_speed_start,
                    speed_stop=args.tap_speed_stop,
                    press_window=args.tap_press_window,
                    smooth_window=args.tap_smooth_window,
                )
            if len(stats) == 0:
                print(f"Skipping {path} (no horizontal runs detected)")
                continue

            ball_px = safe_mean([width[i] for i in range(len(width)) if not np.isnan(width[i])])

            if args.horizontal_style == "hold":
                print(f"\nAnalysis for {dataset_name} (horizontal hold)")
            else:
                print(f"\nAnalysis for {dataset_name} (horizontal tap)")
            print(f"Detected runs: {len(stats)}")
            if ball_px:
                print(f"Mean on-screen ball diameter: {ball_px:.2f} px")
            print()
            if args.horizontal_style == "hold":
                header = "run duration(s) dir accel(px/s^2) accel(ball/s^2) v_terminal(px/s) v_terminal(ball/s)"
                print(header)
                for s in stats:
                    print(
                        f"{s.index:02d} {s.duration_s:8.3f} {s.direction:>5s} "
                        f"{s.acceleration_px:14.2f} {format_val(s.acceleration_units, 2):>16s} "
                        f"{s.terminal_velocity_px:17.2f} {format_val(s.terminal_velocity_units, 2):>20s}"
                    )
                summaries.append(
                    {
                        "name": dataset_name,
                        "count": len(stats),
                        "ball_px": ball_px,
                        "accel_px": mean_abs([s.acceleration_px for s in stats]),
                        "accel_ball": mean_abs([s.acceleration_units for s in stats]),
                        "v_px": mean_abs([s.terminal_velocity_px for s in stats]),
                        "v_ball": mean_abs([s.terminal_velocity_units for s in stats]),
                    }
                )
            else:
                header = (
                    "run press_dur(s) dir thrust(ball/s^2) release(ball/s^2) peak_v(ball/s) glide(s) distance(ball)"
                )
                print(header)
                for s in stats:
                    print(
                        f"{s.index:02d} {s.press_duration_s:8.3f} {s.direction:>5s} "
                        f"{format_val(s.thrust_acceleration_units, 2):>18s} {format_val(s.release_acceleration_units, 2):>17s} "
                        f"{format_val(s.peak_speed_units, 2):>15s} {format_val(s.glide_duration_s, 2):>9s} "
                        f"{format_val(s.distance_units, 2):>15s}"
                    )
                summaries.append(
                    {
                        "name": dataset_name,
                        "count": len(stats),
                        "ball_px": ball_px,
                        "thrust": mean_abs([s.thrust_acceleration_units for s in stats]),
                        "release": mean_abs([s.release_acceleration_units for s in stats]),
                        "peak_v": mean_abs([s.peak_speed_units for s in stats]),
                        "distance": mean_abs([s.distance_units for s in stats]),
                    }
                )

    if len(summaries) > 1:
        print("\nAggregate summary (sorted by ball size):")
        summaries.sort(key=lambda s: (float("inf") if s["ball_px"] is None else s["ball_px"]))
        if args.mode == "vertical":
            print("dataset ball_px(px) apex(px) apex(ball) gravity(px/s^2) gravity(ball/s^2) jumps")
            for s in summaries:
                print(
                    f"{s['name']:20s} "
                    f"{format_val(s['ball_px']):>11s} "
                    f"{format_val(s['apex_px']):>8s} "
                    f"{format_val(s['apex_ball']):>10s} "
                    f"{format_val(s['gravity_px']):>15s} "
                    f"{format_val(s['gravity_ball']):>17s} "
                    f"{s['count']:5d}"
                )
        else:
            if args.horizontal_style == "hold":
                print("dataset ball_px(px) accel(px/s^2) accel(ball/s^2) v(px/s) v(ball/s) runs")
                for s in summaries:
                    print(
                        f"{s['name']:20s} "
                        f"{format_val(s['ball_px']):>11s} "
                        f"{format_val(s['accel_px']):>14s} "
                        f"{format_val(s['accel_ball']):>17s} "
                        f"{format_val(s['v_px']):>9s} "
                        f"{format_val(s['v_ball']):>11s} "
                        f"{s['count']:5d}"
                    )
            else:
                print("dataset ball_px(px) thrust(ball/s^2) release(ball/s^2) peak_v(ball/s) distance(ball) runs")
                for s in summaries:
                    print(
                        f"{s['name']:20s} "
                        f"{format_val(s['ball_px']):>11s} "
                        f"{format_val(s['thrust']):>18s} "
                        f"{format_val(s['release']):>17s} "
                        f"{format_val(s['peak_v']):>15s} "
                        f"{format_val(s['distance']):>14s} "
                        f"{s['count']:5d}"
                    )


if __name__ == "__main__":
    main()
