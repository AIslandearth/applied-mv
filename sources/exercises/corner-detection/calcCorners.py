"""
SimpleCorners, sum/diff heuristic on all detected points
DenseCorners, slot/density filter then sum/diff on dense points only
CurveCorners, convex hull + approxPolyDP, works on any shape

"""
import cv2
import numpy as np
from findEdges import *

    # Global warp
def warpToCorners(gray: np.ndarray, corners: np.ndarray):
    """
    Perspective-warp gray to a rectangle whose size matches the aspect
    ratio of the detected corners (not forced square).

    Returns (warped_gray, M) or None if corners is None.
    """
    if corners is None:
        return None

    tl, tr, br, bl = corners

    # Width = average of top and bottom edge lengths
    w = int((np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2)
    # Height = average of left and right edge lengths
    h = int((np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2)

    if w < 10 or h < 10:
        return None

    dst = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    M   = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(gray, M, (w, h)), M


# Find corners based on sum/diff calc
class SimpleCorners:
    """
    Find 4 corners from detected edge points using sum/diff heuristic.

    Top-left = point where (y + x) is smallest
    Top-right = point where (y - x) is smallest
    Bottom-right = point where (y + x) is largest
    Bottom-left = point where (y - x) is largest

    Parameters
    ys      : 1D int array of row positions
    centers : 1D int array of column (center) positions
    """

    def __init__(self, edgeClass: type):
        self.edgeClass = edgeClass
        self.ys = self.edgeClass.ys
        self.centers = self.edgeClass.centers
        self.diffArr = self.edgeClass.diffArr
        
        # Set by find()
        self.corners = None
        
        self.find()

    def find(self):
        if len(self.ys) < 4:
            return None

        points = np.column_stack([self.ys, self.centers])
        s = points[:, 0] + points[:, 1]   # y + x
        d = points[:, 0] - points[:, 1]   # y - x

        self.corners = np.float32([
            [points[s.argmin()][1], points[s.argmin()][0]],  # TL  (cx, cy)
            [points[d.argmin()][1], points[d.argmin()][0]],  # TR
            [points[s.argmax()][1], points[s.argmax()][0]],  # BR
            [points[d.argmax()][1], points[d.argmax()][0]],  # BL
        ])
        return self.corners

# Find based on slot density and then sum/diff
class DenseCorners:
    """
    Find 4 corners by first filtering to dense slots, then applying
    the sum/diff heuristic on the surviving dense points only.

    The image is divided into a slots x slots grid.  Only points that
    fall in slots with >= slotThreshold % of the maximum slot count
    are kept.  This removes noise and isolated edge fragments before
    corner estimation.

    Parameters
    ys             : 1-D int array of row positions
    centers        : 1-D int array of column positions
    img_shape      : (height, width) of the source image
    slots          : number of grid divisions per axis  (default 4)
    slot_threshold : keep slots with >= this % of max count (default 20)
    """

    def __init__(self, edgeClass: type):
        self.edgeClass = edgeClass
        self.ys = self.edgeClass.ys
        self.centers = self.edgeClass.centers
        self.img_shape = self.edgeClass.diffArr.shape
        self.slots = self.edgeClass.slots
        self.slot_threshold = self.edgeClass.slotThreshold * 0.01
        self.diffArr = self.edgeClass.diffArr
        self.corners = None
        
        self.find()

    def find(self):
        if len(self.ys) < 4:
            return None

        h, w     = self.img_shape
        slot_h   = max(h // self.slots, 1)
        slot_w   = max(w // self.slots, 1)

        # Which slot each point locates
        sy = np.clip(self.ys      // slot_h, 0, self.slots - 1)
        sx = np.clip(self.centers // slot_w, 0, self.slots - 1)

        # Count points per slot, vectorized
        counts = np.zeros((self.slots, self.slots), dtype=np.int32)
        np.add.at(counts, (sy, sx), 1)

        max_count = counts.max()
        if max_count == 0:
            return None

        # Keep only points in slots above the density threshold
        threshold  = max_count * self.slot_threshold
        dense_mask = counts[sy, sx] >= threshold

        dense_ys      = self.ys[dense_mask]
        dense_centers = self.centers[dense_mask]

        if len(dense_ys) < 4:
            return None

        # Sum/diff heuristic on dense points
        points = np.column_stack([dense_ys, dense_centers])
        s = points[:, 0] + points[:, 1]
        d = points[:, 0] - points[:, 1]

        self.corners = np.float32([
            [points[s.argmin()][1], points[s.argmin()][0]],  # TL
            [points[d.argmin()][1], points[d.argmin()][0]],  # TR
            [points[s.argmax()][1], points[s.argmax()][0]],  # BR
            [points[d.argmax()][1], points[d.argmax()][0]],  # BL
        ])
        return self.corners

# Find based on cover outmost corners (convex hull) + simplify w/ approxPolyDP
class CurveCorners:
    """
    Corner finder for curved, irregular, or unknown shapes.

    Instead of assuming 4 corners exist, this class:
      1. Builds a convex hull from all detected edge points
      2. Simplifies the hull with approxPolyDP at adjustable precision
      3. Reports however many corners actually exist (3, 4, 5, ...)
      4. If exactly 4 corners found  -> standard TL/TR/BR/BL ordering
         If not 4 corners            -> returns all hull points as-is
                                        (use for drawing, not warping)

    The hull step is the key difference from SimpleCorners/DenseCorners:
    it traces the outer boundary of the point cloud rather than finding
    the four extreme points, so concave shapes, circles and L-shapes
    are handled correctly.

    Parameters
    ys          : 1-D int array of row positions
    centers     : 1-D int array of column positions
    epsilon_frac: approxPolyDP epsilon as fraction of hull perimeter.
                  smaller = more corners kept  (default 0.02)
                  larger  = more simplification, fewer corners
    """

    def __init__(self, edgeClass: type):
        self.edgeClass = edgeClass()
        self.ys           = self.edgeClass.ys
        self.centers      = self.edgeClass.centers
        self.epsilon_frac = self.edgeClass.epsilon_frac
        self.corners      = None # set by find(), may have <> 4 points
        self.hull         = None # full convex hull points
        self.n_corners    = 0 # how many corners were found
        self.diffArr = self.edgeClass.diffArr
        self.find()
        
    def find(self):
        if len(self.ys) < 4:
            return None

        # Build (y, x) point array, cv2 convention is (col, row)
        pts = np.column_stack([self.centers.astype(np.float32),self.ys.astype(np.float32),]).reshape(-1, 1, 2)

        # Convex hull — outer boundary of all detected points
        hull = cv2.convexHull(pts)
        self.hull = hull.reshape(-1, 2)

        # Simplify hull to polygon corners
        epsilon = self.epsilon_frac * cv2.arcLength(hull, closed=True)
        approx = cv2.approxPolyDP(hull, epsilon, closed=True)
        approx_pts = approx.reshape(-1, 2).astype(np.float32)

        self.n_corners = len(approx_pts)

        if self.n_corners == 4:
            # Order as TL, TR, BR, BL — safe to pass to warp_to_corners
            self.corners = self._order_quad(approx_pts)
        else:
            # Return all hull corners
            self.corners = approx_pts

        return self.corners

    def is_quad(self):
        return self.n_corners == 4

    def draw_hull(self, out: np.ndarray, color: tuple = (0, 255, 180), thickness: int = 1):
        # Draw the full convex hull outline on a BGR image.
        if self.hull is None:
            return out
        
        pts = self.hull.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)
        
        return out

    def draw_corners(self, out: np.ndarray, radius: int = 8):
        # Draw all detected corners
        if self.corners is None:
            return out
        for pt in self.corners:
            cv2.circle(out, (int(pt[0]), int(pt[1])), radius, (255, 255, 255), -1)
            cv2.circle(out, (int(pt[0]), int(pt[1])), radius, (0, 0, 0), 2)
        
        return out

    @staticmethod
    def _order_quad(pts: np.ndarray):
        # Order 4 points as TL, TR, BR, BL
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1).ravel()
        return np.float32([
            pts[s.argmin()], # TL
            pts[d.argmin()], # TR
            pts[s.argmax()], # BR
            pts[d.argmax()], # BL
        ])