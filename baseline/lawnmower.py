"""
Naive lawnmower flight pattern -- what a human operator with no smart
routing would fly: divide the map into one horizontal band per drone,
sweep each band back and forth (boustrophedon), full stop.

Deterministic, stateless, pure function of t -- this is the "dumb"
baseline the Phase 6 checkpoint compares engine.orchestrator against
(coverage curve), and what /baseline/nearest_first.py reuses for the
Phase 7 evaluation's drone behaviour. It uses the exact same physical
speed assumption as the orchestrator (DRONE_SPEED_M_PER_MIN) so the
comparison is apples-to-apples -- same hardware, different flight logic.
"""

from __future__ import annotations

import math
from typing import Tuple

from config import params

ROW_SPACING_M = 200.0  # [ASSUMPTION] vertical spacing between sweep passes
                        # within a drone's band -- dense enough for
                        # reasonable coverage without excessive passes


def lawnmower_position(drone_index: int, n_drones: int, t_seconds: float, map_size: float,
                        speed_m_per_min: float = params.DRONE_SPEED_M_PER_MIN) -> Tuple[float, float]:
    band_h = map_size / n_drones
    band_y0 = drone_index * band_h
    n_rows = max(1, math.ceil(band_h / ROW_SPACING_M))

    t_min = t_seconds / 60.0
    cycle_min = (n_rows * map_size) / speed_m_per_min  # total time for one full band sweep
    if cycle_min <= 0:
        return (0.0, band_y0 + band_h / 2)

    s = (t_min % cycle_min) / cycle_min * n_rows  # 0..n_rows, continuous
    row = min(int(s), n_rows - 1)
    frac = s - row

    x = map_size * frac if row % 2 == 0 else map_size * (1 - frac)
    y = min(band_y0 + (row + 0.5) * ROW_SPACING_M, band_y0 + band_h - 1e-6)
    x = min(max(x, 0.0), map_size)
    y = min(max(y, 0.0), map_size)
    return (x, y)


if __name__ == "__main__":
    for t in [0, 30, 300, 3000]:
        print(f"t={t}s -> drone0: {lawnmower_position(0, 4, t, params.MAP_SIZE_M)}, "
              f"drone2: {lawnmower_position(2, 4, t, params.MAP_SIZE_M)}")
