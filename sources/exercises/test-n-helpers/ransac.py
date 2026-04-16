def ransac_lines(pts: np.ndarray, iterations: int = 100,
                 threshold: float = 2.0, min_inliers: int = 30) -> list:
    lines = []
    remaining = pts.copy()

    while len(remaining) >= 2:
        best_inliers = []

        for _ in range(iterations):
            # pick 2 random points
            idx = np.random.choice(len(remaining), 2, replace=False)
            p1, p2 = remaining[idx]

            # distance from all points to this line
            inliers = _inliers(remaining, p1, p2, threshold)

            if len(inliers) > len(best_inliers):
                best_inliers = inliers

        if len(best_inliers) < min_inliers:
            break

        lines.append(best_inliers)
        # remove inliers, repeat on remaining points
        mask = np.ones(len(remaining), dtype=bool)
        mask[best_inliers] = False
        remaining = remaining[mask]

    return lines


def _inliers(pts: np.ndarray, p1, p2, threshold: float) -> np.ndarray:
    # perpendicular distance from each point to line p1->p2
    d = np.abs(np.cross(p2 - p1, p1 - pts)) / np.linalg.norm(p2 - p1)
    return np.where(d < threshold)[0]