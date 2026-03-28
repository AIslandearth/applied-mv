"""
geometry.py
-----------
Pure math utilities for line detection:
  - slope/intercept calculation
  - pairwise intersection finding
"""

import itertools


def to_slope_intercept(x1: float, y1: float, x2: float, y2: float) -> tuple[float | None, float]:
    """
    Return (k, b) for the line y = kx + b through (x1,y1)→(x2,y2).
    For vertical lines k=None and b holds the x-coordinate.
    """
    if abs(x2 - x1) < 1e-8:
        return None, float(x1)
    k = (y2 - y1) / (x2 - x1)
    return k, y1 - k * x1


def line_intersections(slopes_intercepts: list[tuple]) -> list[tuple]:
    """
    Given a list of (k, b) pairs, return all pairwise intersection points
    as (x, y, line_index_i, line_index_j).  Indices are 1-based.
    Parallel and double-vertical pairs are skipped.
    """
    intersections = []
    for (i, (k1, b1)), (j, (k2, b2)) in itertools.combinations(
        enumerate(slopes_intercepts), 2
    ):
        if k1 is not None and k2 is not None:
            if abs(k1 - k2) < 1e-8:          # parallel
                continue
            x = (b2 - b1) / (k1 - k2)
            y = k1 * x + b1
        elif k1 is None and k2 is not None:   # line 1 vertical
            x = b1
            y = k2 * x + b2
        elif k2 is None and k1 is not None:   # line 2 vertical
            x = b2
            y = k1 * x + b1
        else:                                  # both vertical
            continue
        intersections.append((x, y, i + 1, j + 1))
    return intersections
