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


def load_json_recording(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = json.loads(path.read_text())
    t = np.array([float(row[0]) for row in raw])
    y = np.array([float(row[2]) for row in raw])
    width = np.full_like(y, np.nan, dtype=float)
    return t, y, width


def load_csv_recording(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    times: List[float] = []
    ys: List[float] = []
    widths: List[float] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row["Time_ms"]))
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
    y = np.array(ys, dtype=float)
    width = np.array(widths, dtype=float)
    return t, y, width


def load_recording(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if path.suffix.lower() == ".json":
        t, y, width = load_json_recording(path)
    elif path.suffix.lower() == ".csv":
        t, y, width = load_csv_recording(path)
    else:
        raise ValueError(f"Unsupported file format: {path}")

    # Drop duplicate timestamps by keeping the last sample at each time.
    dedup_mask = np.concatenate((np.diff(t) != 0, [True]))
    t = t[dedup_mask]
    y = y[dedup_mask]
    width = width[dedup_mask]
    return t, y, width


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
    parser.add_argument(
        "--skip-plots", action="store_true", help="disable PNG output (faster)"
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
        t, y, width = load_recording(path)
        if len(t) == 0:
            print(f"Skipping {path} (empty recording)")
            continue

        stats, mean_gravity = analyze_jumps(t, y, width)
        if len(stats) == 0:
            print(f"Skipping {path} (no jumps detected)")
            continue

        dataset_name = path.stem
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
        header = (
            "jump duration(s) apex_y apex_height takeoff_v(px/s) gravity(px/s^2) apex_t(s)"
        )
        print(header)
        for s in stats:
            print(
                f"{s.index:02d} {s.duration_s:8.3f} {s.apex_y:7.2f} {s.apex_height:11.2f} "
                f"{s.takeoff_velocity:14.2f} {s.gravity:15.2f} {s.apex_time_s:9.3f}"
            )

        summaries.append(
            {
                "name": dataset_name,
                "jumps": len(stats),
                "baseline": baseline,
                "ball_px": ball_px,
                "apex_px": safe_mean([s.apex_height for s in stats]),
                "apex_ball": safe_mean([s.apex_height_units for s in stats]),
                "gravity_px": mean_gravity,
                "gravity_ball": safe_mean([s.gravity_units for s in stats]),
                "takeoff_px": safe_mean([s.takeoff_velocity for s in stats]),
                "takeoff_ball": safe_mean([s.takeoff_velocity_units for s in stats]),
            }
        )

    if len(summaries) > 1:
        print("\nAggregate summary (sorted by ball size):")
        summaries.sort(
            key=lambda s: (float("inf") if s["ball_px"] is None else s["ball_px"])
        )
        print(
            "dataset ball_px(px) apex(px) apex(ball) gravity(px/s^2) gravity(ball/s^2) jumps"
        )
        for s in summaries:
            print(
                f"{s['name']:20s} "
                f"{format_val(s['ball_px']):>11s} "
                f"{format_val(s['apex_px']):>8s} "
                f"{format_val(s['apex_ball']):>10s} "
                f"{format_val(s['gravity_px']):>15s} "
                f"{format_val(s['gravity_ball']):>17s} "
                f"{s['jumps']:5d}"
            )


if __name__ == "__main__":
    main()
