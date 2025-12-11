# Bonk.io Physics Notes

This document will grow as we collect more recordings with `physics-extractor/1.js` and crunch the data in `analyze_jumps.py`.

## Dataset `0-jumps-up-tap`

- Scenario: player on a flat, non-bouncy platform, tapping the UP key roughly once every ~1.5 s.
- Recording cadence ≈120 Hz, but each frame is duplicated (exact same timestamp/position pair). The analysis script deduplicates by keeping the last sample per timestamp.
- Resting Y coordinate: **394.48 px**. Horizontal position is constant (we stood still).
- Detected 11 clean jumps. Six reach **11.0 px** above the floor, five reach **10.23 px**. The shorter group likely corresponds to under-held taps (human timing error). Plot files live in `physics-extractor/analysis/`.

| Jump group | Apex height (px) | Apex time (s) | Total airtime (s) | Takeoff velocity (px/s) | Gravity (px/s²) |
|------------|-----------------|---------------|-------------------|-------------------------|-----------------|
| “Full” tap (6 hops) | 11.00 | 0.50 | ≈1.00 | -40.4 | 81.0 |
| Short tap (5 hops) | 10.23 | 0.48 | ≈0.97 | -39.3 | 81.0 |

Gravity is very stable across all jumps: **~81 px/s²** downward. The normalized jump curves overlay perfectly, indicating a simple constant-acceleration parabola (no measurable air drag in this test).

## Known limitations / next actions

- **Input consistency**: human taps introduce ~0.8 px differences in apex height. Use an automated key tapper (e.g., ydotool or a PIXI ticker helper inside the injector) to nail single-frame presses.
- **Pixel scaling**: 81 px/s² depends on the camera zoom; we need to log a conversion factor (player diameter in px, block widths, etc.) to express gravity in world units that survive zoom changes.
- **Future experiments**: test multi-frame holds to map jump strength vs. hold time, record horizontal motion to measure acceleration/drag, and gather data on bouncy surfaces.

Update this document as new recordings are analyzed so we build a full physics profile over time.
