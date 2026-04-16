def hough_lines(edges: np.ndarray, threshold: int = 50) -> list:
    h, w = edges.shape
    diagonal = int(np.sqrt(h**2 + w**2))
    thetas = np.deg2rad(np.arange(-90, 90))
    rhos = np.arange(-diagonal, diagonal)

    # accumulator array
    acc = np.zeros((len(rhos), len(thetas)), dtype=np.int32)

    # vote
    ys, xs = np.where(edges > 0)
    for x, y in zip(xs, ys):
        for t, theta in enumerate(thetas):
            rho = int(x * np.cos(theta) + y * np.sin(theta))
            r = rho + diagonal  # shift to positive index
            acc[r, t] += 1

    # find peaks above threshold
    lines = []
    for r, t in zip(*np.where(acc > threshold)):
        rho = rhos[r]
        theta = thetas[t]
        lines.append((rho, theta))

    return lines