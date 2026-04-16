import cv2
import numpy as np

def hough_lines_p(edges: np.ndarray, threshold: int = 50,
                  min_length: int = 30, max_gap: int = 10) -> list:
    h, w = edges.shape
    diagonal = int(np.sqrt(h**2 + w**2))
    thetas = np.deg2rad(np.arange(-90, 90))

    acc = np.zeros((2 * diagonal, len(thetas)), dtype=np.int32)

    # get all edge points and shuffle — the "probabilistic" part
    ys, xs = np.where(edges > 0)
    pts = list(zip(xs, ys))
    np.random.shuffle(pts)

    for x, y in pts:
        for t, theta in enumerate(thetas):
            rho = int(x * np.cos(theta) + y * np.sin(theta)) + diagonal
            acc[rho, t] += 1

            # if this bin hits threshold, trace the line segment
            if acc[rho, t] == threshold:
                lines = _trace_line(edges, rho - diagonal, theta,
                                    min_length, max_gap)
                yield from lines


def _trace_line(edges, rho, theta, min_length, max_gap):
    h, w = edges.shape
    lines = []
    gap = 0
    start = None
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    # walk along the line direction
    for i in range(-max(h, w), max(h, w)):
        x = int(rho * cos_t - i * sin_t)
        y = int(rho * sin_t + i * cos_t)

        if 0 <= x < w and 0 <= y < h and edges[y, x] > 0:
            if start is None:
                start = (x, y)
            gap = 0
            end = (x, y)
        else:
            gap += 1
            if gap > max_gap and start is not None:
                if np.sqrt((end[0]-start[0])**2 + (end[1]-start[1])**2) >= min_length:
                    lines.append((start[0], start[1], end[0], end[1]))
                start = None

    return lines