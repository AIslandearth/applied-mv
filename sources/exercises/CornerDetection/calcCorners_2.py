import cv2
import numpy as np
from findEdges_2 import EdgeDetector


# ── Warp ──────────────────────────────────────────────────────────────────────

def warpToCorners(gray: np.ndarray, corners: np.ndarray | None) -> np.ndarray | None:
    """Perspective-warp *gray* to a rectangle fitted to *corners*, or None."""
    if corners is None: return None
    tl, tr, br, bl = corners
    w = int((np.linalg.norm(tr-tl) + np.linalg.norm(br-bl)) / 2)
    h = int((np.linalg.norm(bl-tl) + np.linalg.norm(br-tr)) / 2)
    if w < 10 or h < 10: return None
    dst = np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
    return cv2.warpPerspective(gray, cv2.getPerspectiveTransform(corners, dst), (w, h))


# ── Corner base ───────────────────────────────────────────────────────────────

class _CornerBase:
    corners: np.ndarray | None

    def __init__(self) -> None:
        self.corners = None

    def draw_corners(self, out: np.ndarray, r: int = 10) -> np.ndarray:
        for pt in (self.corners if self.corners is not None else []):
            cv2.circle(out, (int(pt[0]), int(pt[1])), r, (255,255,255), -1)
            cv2.circle(out, (int(pt[0]), int(pt[1])), r, (0,0,0), 2)
        return out

    @staticmethod
    def _quad(ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
        """Sum/diff heuristic → TL, TR, BR, BL as (cx, cy) float32."""
        pts = np.column_stack([ys, xs])          # col-0 = row, col-1 = col
        s, d = pts[:,0] + pts[:,1], pts[:,0] - pts[:,1]
        idx  = [s.argmin(), d.argmin(), s.argmax(), d.argmax()]
        return np.float32([[pts[i,1], pts[i,0]] for i in idx])


# ── Corner finders ────────────────────────────────────────────────────────────

class SimpleCorners(_CornerBase):
    """Sum/diff heuristic on every detected edge point."""

    def __init__(self, edge: EdgeDetector) -> None:
        super().__init__()
        if len(edge.ys) >= 4:
            self.corners = self._quad(edge.ys, edge.centers)


class DenseCorners(_CornerBase):
    """
    Keep only points in dense grid cells, then apply the sum/diff heuristic.

    Divides the image into a (slots × slots) grid; discards cells whose
    point count is below *slotThreshold* % of the busiest cell.
    """

    def __init__(self, edge: EdgeDetector) -> None:
        super().__init__()
        ys, xs = edge.ys, edge.centers
        if len(ys) < 4: return

        h, w   = edge.diffArr.shape[:2]
        slots  = edge.slots
        sh, sw = max(h // slots, 1), max(w // slots, 1)

        sy = np.clip(ys // sh, 0, slots-1)
        sx = np.clip(xs // sw, 0, slots-1)

        counts = np.zeros((slots, slots), dtype=np.int32)
        np.add.at(counts, (sy, sx), 1)

        max_c = counts.max()
        if max_c == 0: return

        mask = counts[sy, sx] >= max_c * (edge.slotThreshold / 100)
        if mask.sum() < 4: return

        self.corners = self._quad(ys[mask], xs[mask])


class CurveCorners(_CornerBase):
    """
    Convex hull + approxPolyDP — handles any shape.

    Finds however many corners exist; if exactly 4, orders them TL/TR/BR/BL
    and they are safe to pass to warpToCorners().
    """

    def __init__(self, edge: EdgeDetector, epsilon_frac: float = 0.02) -> None:
        super().__init__()
        self.hull, self.n_corners = None, 0
        ys, xs = edge.ys, edge.centers
        if len(ys) < 4: return

        pts  = np.column_stack([xs, ys]).astype(np.float32).reshape(-1, 1, 2)
        hull = cv2.convexHull(pts)
        self.hull = hull.reshape(-1, 2)

        approx = cv2.approxPolyDP(hull, epsilon_frac * cv2.arcLength(hull, True), True)
        pts4   = approx.reshape(-1, 2).astype(np.float32)

        self.n_corners = len(pts4)
        self.corners   = self._quad(pts4[:,1], pts4[:,0]) if self.n_corners == 4 else pts4

    def is_quad(self) -> bool:
        return self.n_corners == 4

    def draw_hull(self, out: np.ndarray,
                  color: tuple = (0,255,180), thickness: int = 1) -> np.ndarray:
        if self.hull is not None:
            cv2.polylines(out, [self.hull.astype(np.int32).reshape(-1,1,2)],
                          True, color, thickness, cv2.LINE_AA)
        return out
