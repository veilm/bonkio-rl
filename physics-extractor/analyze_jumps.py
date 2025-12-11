#!/usr/bin/env python3
"""
Analyze Bonk.io jump recordings produced by physics-extractor/1.js.

Usage:
    python analyze_jumps.py data/0-jumps-up-tap.json

The script prints summary statistics and emits plots inside
physics-extractor/analysis/.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

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


def load_recording(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    raw = json.loads(path.read_text())
    t = np.array([float(row[0]) for row in raw])
    y = np.array([float(row[2]) for row in raw])

    # Drop duplicate timestamps by keeping the last sample at each time.
    dedup_mask = np.concatenate((np.diff(t) != 0, [True]))
    t = t[dedup_mask]
    y = y[dedup_mask]
    return t, y


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


def analyze_jumps(t: np.ndarray, y: np.ndarray) -> Tuple[List[JumpStats], float]:
    baseline = float(np.max(y))
    segments = detect_airborne_segments(t, y, baseline)
    stats: List[JumpStats] = []
    for idx, (start, end) in enumerate(segments):
        seg_slice = slice(start, end + 1)
        seg_t = (t[seg_slice] - t[start]) / 1000.0  # seconds relative to takeoff
        seg_y = y[seg_slice]

        if len(seg_t) < 5:
            continue
        a, b, c = fit_parabola(seg_t, seg_y)
        gravity = 2 * a  # because y = y0 + v0*t + 0.5*g*t^2
        apex_time = float(-b / (2 * a))
        apex_y = float(np.polyval([a, b, c], apex_time))
        apex_height = baseline - apex_y

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("recording", type=Path, help="path to JSON recording")
    args = parser.parse_args()

    t, y = load_recording(args.recording)
    stats, mean_gravity = analyze_jumps(t, y)
    dataset_name = args.recording.stem
    out_dir = Path("physics-extractor") / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline = float(np.max(y))
    plot_overview(t, y, stats, baseline, out_dir, dataset_name)
    plot_normalized_jumps(t, y, stats, baseline, out_dir, dataset_name)
    plot_apexes(stats, out_dir, dataset_name)

    print(f"Analysis for {dataset_name}")
    print(f"Detected jumps: {len(stats)}")
    print(f"Mean gravity (px/s^2): {mean_gravity:.2f}")
    print(f"Rest height y: {baseline:.2f}")
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


if __name__ == "__main__":
    main()
