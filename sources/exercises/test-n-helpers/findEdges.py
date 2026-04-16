"""
StepDiff, stepsize difference array -> binary -> rising/falling/center
Derivative, smooth -> d1 -> d2 zero crossings -> center
SimpleCorners, sum/diff heuristic on all detected points
DenseCorners, slot/density filter then sum/diff on dense points only
CurveCorners, convex hull + approxPolyDP, works on any shape

"""

import cv2
import numpy as np

# Step width based difference, then to bin
class StepDiff:
    """
    Edge detector using stepsize finite difference -> binary -> scan lines.

    Algorithm
    Build diff array:
        diff_x[y, x] = |gray[y, x+step] - gray[y, x-step]|   (horizontal)
        diff_y[y, x] = |gray[y+step, x] - gray[y-step, x]|   (vertical)
        diff[y, x]   = max(diff_x, diff_y) > threshold  -> 0 or 1

    Scan every row AND every column of the binary diff array.
       In each 1-D scan:
         - A run of 1s = one detected edge
         - outer  = first 1  (rising edge)
         - inner  = last  1  (falling edge)
         - center = (outer + inner) // 2

    Parameters
    gray      : grayscale uint8 image
    threshold : minimum difference to count as an edge (default 30)
    step      : finite-difference step size in pixels  (default 2)
    """

    def __init__(self, gray: np.ndarray, diffThreshold: int = 30, step: int = 2, slots: int = 1, slotThreshold: int = 20):
        self.gray = gray
        self.threshold = diffThreshold
        self.step = step
        self.slots = slots
        self.slotThreshold = slotThreshold
        
        # Results — populated by detect()
        self.diffArr = None
        self.ys       = np.array([], dtype=np.int32)
        self.xs       = np.array([], dtype=np.int32)   # column-scan row positions
        self.outers   = np.array([], dtype=np.int32)
        self.inners   = np.array([], dtype=np.int32)
        self.centers  = np.array([], dtype=np.int32)
        self.widths   = np.array([], dtype=np.int32)
        self.detectDiff()
    
    def detectDiff(self):
        self._build_diff()

        h, w = self.diffArr.shape
        row_ys      = []
        row_outers  = []
        row_inners  = []
        col_ys      = []
        col_outers  = []
        col_inners  = []

        # Scan rows
        for y in range(h):
            result = self._scan_line(self.diffArr[y, :])
            if result is not None:
                o, i = result
                row_ys.append(np.full(len(o), y, dtype=np.int32))
                row_outers.append(o)
                row_inners.append(i)

        # Scan columns
        for x in range(w):
            result = self._scan_line(self.diffArr[:, x])
            if result is not None:
                o, i = result
                # o = row positions for column scan
                col_ys.append(o)
                col_outers.append(np.full(len(o), x, dtype=np.int32))
                col_inners.append(np.full(len(o), x, dtype=np.int32))

        # Combine rows
        if row_ys:
            all_ys = np.concatenate(row_ys)
            all_outers = np.concatenate(row_outers)
            all_inners = np.concatenate(row_inners)
        else:
            all_ys = all_outers = all_inners = np.array([], dtype=np.int32)

        self.ys = all_ys
        self.outers = all_outers
        self.inners = all_inners
        self.centers = (all_outers + all_inners) // 2
        self.widths = all_inners - all_outers
        
        return self

    def draw(self, circle_r: int = 3):
        out = cv2.cvtColor(self.gray, cv2.COLOR_GRAY2BGR)
        if len(self.ys) > 0:
            for y, c in zip(self.ys, self.centers):
                cv2.circle(out, (int(c), int(y)), circle_r, (0, 0, 255), -1)
        
        return out

    def draw_corners(self, out: np.ndarray,
                     corners: np.ndarray | None) -> np.ndarray:
        if corners is None:
            return out
        
        colors = [(255, 255, 255)] * 4   # white
        for i, (cx, cy) in enumerate(corners):
            cv2.circle(out, (int(cx), int(cy)), 10, colors[i], -1)
            cv2.circle(out, (int(cx), int(cy)), 10, (0, 0, 0),   2)
        
        return out

    def _build_diff(self):
        s = self.step
        g = self.gray.astype(np.int16)

        # Horizontal diff
        diff_x = np.abs(g[:, s:] - g[:, :-s])
        diff_x = np.pad(diff_x, ((0, 0), (s // 2, s - s // 2)), mode='constant')

        # Vertical diff
        diff_y = np.abs(g[s:, :] - g[:-s, :])
        diff_y = np.pad(diff_y, ((s // 2, s - s // 2), (0, 0)), mode='constant')

        # Align shapes after padding
        h = min(diff_x.shape[0], diff_y.shape[0])
        w = min(diff_x.shape[1], diff_y.shape[1])
        diff_x = diff_x[:h, :w]
        diff_y = diff_y[:h, :w]

        self.diffArr = (np.maximum(diff_x, diff_y) > self.threshold).astype(np.uint8)

    @staticmethod
    def _scan_line(line: np.ndarray):
        """
        Find all runs of 1s in a binary 1-D array.
        Returns (outers, inners) as int32 arrays, or None if no runs.
        outer = index of first 1, inner = index of last 1 in each run.
        """
        d      = np.diff(line.astype(np.int8))
        starts = np.where(d ==  1)[0] + 1 # +1: diff index is one behind
        ends   = np.where(d == -1)[0]

        # Handle edge cases, line starts or ends inside a run
        if len(line) > 0 and line[0] == 1:
            starts = np.concatenate([[0], starts])
        if len(line) > 0 and line[-1] == 1:
            ends = np.concatenate([ends, [len(line) - 1]])

        n = min(len(starts), len(ends))
        if n == 0:
            return None

        # Keep only valid pairs
        mask   = starts[:n] < ends[:n]
        starts = starts[:n][mask]
        ends   = ends[:n][mask]

        if len(starts) == 0:
            return None

        return starts.astype(np.int32), ends.astype(np.int32)


# Derivative based edge detection, first der, second der and zero crossing
class Derivative:
    """
    Edge detector using 1st and 2nd derivative zero-crossing method.

    Algorithm (per scan-line profile)
    1. Smooth profile with 3-tap Gaussian [0.25, 0.5, 0.25]
    2. d1 = np.diff(profile) 1st derivative
       d2 = np.diff(d1) 2nd derivative
    3. Rising edge top = first d1 sample above threshold (refined by d2 +→− zero crossing if found)
    4. Falling edge top = last d1 sample above threshold (refined by d2 −→+ zero crossing if found)
    5. Center = midpoint of rising/falling (refined by parabolic peak of d1 if inside band)

    Scans both rows and columns for better coverage of skewed lines.

    Parameters
    threshold : minimum d1 value to count as edge (default 15)
    step      : difference step for gradient (default 2)
    blur      : Gaussian blur kernel before scanning (0 = off, must be odd)
    """
    
    def __init__(self, gray: np.ndarray, threshold: int = 15, step: int = 2, blur: int = 3):
        self.gray      = gray
        self.threshold = threshold
        self.step      = step
        self.blur      = blur if blur % 2 == 1 else blur + 1

        self.ys      = np.array([], dtype=np.int32)
        self.outers  = np.array([], dtype=np.int32)
        self.inners  = np.array([], dtype=np.int32)
        self.centers = np.array([], dtype=np.int32)
        self.widths  = np.array([], dtype=np.int32)
        
        self.detectDerivative()

    # Derivative
    def detectDerivative(self):
        src = (cv2.GaussianBlur(self.gray, (self.blur, self.blur), 0) if self.blur > 1 else self.gray)

        h, w = src.shape
        row_ys = [];      row_outers = [];  row_inners = []
        col_ys = [];      col_outers = [];  col_inners = []

        # Scan rows
        for y in range(h):
            result = self._scan_profile(src[y, :].astype(np.float32))
            if result is not None:
                o, i = result
                row_ys.append(np.full(len(o), y, dtype=np.int32))
                row_outers.append(o)
                row_inners.append(i)

        # Scan columns
        for x in range(w):
            result = self._scan_profile(src[:, x].astype(np.float32))
            if result is not None:
                oo, ii = result
                col_ys.append(oo)
                col_outers.append(np.full(len(oo), x, dtype=np.int32))
                col_inners.append(np.full(len(oo), x, dtype=np.int32))

        if row_ys:
            all_ys     = np.concatenate(row_ys)
            all_outers = np.concatenate(row_outers)
            all_inners = np.concatenate(row_inners)
        else:
            all_ys = all_outers = all_inners = np.array([], dtype=np.int32)

        self.ys      = all_ys
        self.outers  = all_outers
        self.inners  = all_inners
        self.centers = (all_outers + all_inners) // 2
        self.widths  = all_inners - all_outers

        return self

    def draw(self, circle_r: int = 3):
        out = cv2.cvtColor(self.gray, cv2.COLOR_GRAY2BGR)
        if len(self.ys) > 0:
            for y, c in zip(self.ys, self.centers):
                cv2.circle(out, (int(c), int(y)), circle_r, (0, 0, 255), -1)
        return out

    def draw_corners(self, out: np.ndarray, corners: np.ndarray | None):
        if corners is None:
            return out
        for cx, cy in corners:
            cv2.circle(out, (int(cx), int(cy)), 10, (255, 255, 255), -1)
            #cv2.circle(out, (int(cx), int(cy)), 10, (0,   0,   0),    2)
        return out

    def _scan_profile(self, profile: np.ndarray):
        """
        Find all edges in a 1-D intensity profile via d1/d2 method.
        Returns (outers, inners) as int32 arrays, or None.
        """
        smooth_k = np.array([0.25, 0.50, 0.25], dtype=np.float32)
        p   = np.convolve(profile, smooth_k, mode='same')
        d1  = np.diff(p)
        nd1 = len(d1)
        margin = 2

        if nd1 <= 2 * margin:
            return None

        d2 = np.diff(d1)

        outers_list = []
        inners_list = []

        # Walk through d1 looking for above-threshold bands
        i = margin
        while i < nd1 - margin:
            if d1[i] < self.threshold:
                i += 1
                continue

            # Found start of a band, find the full extent
            band_start = i
            while i < nd1 - margin and d1[i] >= self.threshold:
                i += 1
            band_end = i - 1

            # Rising edge top: d2 + to - left of band_start
            outer_f = self._refine_zero_crossing(d2, band_start, pos_to_neg=True,
                                      search_range=4, direction='left')
            outer = int(np.clip(round(outer_f + 0.5), 0, len(profile) - 1))

            # Falling edge top: d2 - to + right of band_end
            inner_f = self._refine_zero_crossing(d2, band_end, pos_to_neg=False,
                                      search_range=4, direction='right')
            inner = int(np.clip(round(inner_f + 0.5), 0, len(profile) - 1))

            # Fallback: if refinement collapsed outer >= inner, use raw band
            if inner <= outer:
                outer = int(np.clip(band_start, 0, len(profile) - 1))
                inner = int(np.clip(band_end + 1, 0, len(profile) - 1))

            if inner > outer:
                outers_list.append(outer)
                inners_list.append(inner)

        if not outers_list:
            return None

        return (np.array(outers_list, dtype=np.int32), np.array(inners_list, dtype=np.int32))

    @staticmethod
    def _refine_zero_crossing(d2: np.ndarray, near: int, pos_to_neg: bool, search_range: int, direction: str = 'left') -> float:
        """
        Find nearest d2 zero crossing of given polarity "near".
        direction='left', search leftward  (for rising edge top)
        direction='right', search rightward (for falling edge top)
        Returns subpixel d2 index, or float(near) if none found.
        """
        if direction == 'left':
            lo = max(0, near - search_range)
            hi = min(len(d2) - 1, near + 1)
            indices = range(lo, hi)
        else:
            lo = max(0, near - 1)
            hi = min(len(d2) - 1, near + search_range)
            indices = range(lo, hi)

        for j in indices:
            a, b = d2[j], d2[j + 1]
            if a == b:
                continue
            if (a > 0) != (b > 0) and ((a > 0) == pos_to_neg):
                t = -a / (b - a)
                return j + t
        return float(near)

    @staticmethod
    def _parabolic(sig: np.ndarray, i: int):
        if i <= 0 or i >= len(sig) - 1:
            return float(i)
        a, b, c = sig[i - 1], sig[i], sig[i + 1]
        denom = 2.0 * (2.0 * b - a - c)
        if abs(denom) < 1e-8:
            return float(i)
        return float(i) + (a - c) / denom

# # ──────────────────────────────────────────────────────────────────────────────
# # Corner finders
# # ──────────────────────────────────────────────────────────────────────────────

# class SimpleCorners:
#     """
#     Find 4 corners from detected edge points using sum/diff heuristic.

#     Top-left     = point where (y + x) is smallest
#     Top-right    = point where (y - x) is smallest
#     Bottom-right = point where (y + x) is largest
#     Bottom-left  = point where (y - x) is largest

#     Parameters
#     ----------
#     ys      : 1-D int array of row positions
#     centers : 1-D int array of column (center) positions
#     """

#     def __init__(self, ys: np.ndarray, centers: np.ndarray):
#         self.ys      = ys
#         self.centers = centers
#         self.corners = None   # set by find()

#     def find(self) -> np.ndarray | None:
#         if len(self.ys) < 4:
#             return None

#         points = np.column_stack([self.ys, self.centers])
#         s = points[:, 0] + points[:, 1]   # y + x
#         d = points[:, 0] - points[:, 1]   # y - x

#         self.corners = np.float32([
#             [points[s.argmin()][1], points[s.argmin()][0]],  # TL
#             [points[d.argmin()][1], points[d.argmin()][0]],  # TR
#             [points[s.argmax()][1], points[s.argmax()][0]],  # BR
#             [points[d.argmax()][1], points[d.argmax()][0]],  # BL
#         ])
#         return self.corners


# class DenseCorners:
#     """
#     Find 4 corners by first filtering to dense slots, then applying
#     the sum/diff heuristic on the surviving dense points only.

#     The image is divided into a slots x slots grid.  Only points that
#     fall in slots with >= slotThreshold % of the maximum slot count
#     are kept.  This removes noise and isolated edge fragments before
#     corner estimation.

#     Parameters
#     ----------
#     ys             : 1-D int array of row positions
#     centers        : 1-D int array of column positions
#     img_shape      : (height, width) of the source image
#     slots          : number of grid divisions per axis  (default 4)
#     slot_threshold : keep slots with >= this % of max count (default 20)
#     """

#     def __init__(self,
#                  ys:             np.ndarray,
#                  centers:        np.ndarray,
#                  img_shape:      tuple[int, int],
#                  slots:          int   = 4,
#                  slot_threshold: float = 20.0):
#         self.ys             = ys
#         self.centers        = centers
#         self.img_shape      = img_shape
#         self.slots          = slots
#         self.slot_threshold = slot_threshold * 0.01
#         self.corners        = None

#     def find(self) -> np.ndarray | None:
#         if len(self.ys) < 4:
#             return None

#         h, w     = self.img_shape
#         slot_h   = max(h // self.slots, 1)
#         slot_w   = max(w // self.slots, 1)

#         # Which slot does each point fall in?
#         sy = np.clip(self.ys      // slot_h, 0, self.slots - 1)
#         sx = np.clip(self.centers // slot_w, 0, self.slots - 1)

#         # Count points per slot — vectorized
#         counts = np.zeros((self.slots, self.slots), dtype=np.int32)
#         np.add.at(counts, (sy, sx), 1)

#         max_count = counts.max()
#         if max_count == 0:
#             return None

#         # Keep only points in slots above the density threshold
#         threshold  = max_count * self.slot_threshold
#         dense_mask = counts[sy, sx] >= threshold

#         dense_ys      = self.ys[dense_mask]
#         dense_centers = self.centers[dense_mask]

#         if len(dense_ys) < 4:
#             return None

#         # Sum/diff heuristic on dense points
#         points = np.column_stack([dense_ys, dense_centers])
#         s = points[:, 0] + points[:, 1]
#         d = points[:, 0] - points[:, 1]

#         self.corners = np.float32([
#             [points[s.argmin()][1], points[s.argmin()][0]],  # TL
#             [points[d.argmin()][1], points[d.argmin()][0]],  # TR
#             [points[s.argmax()][1], points[s.argmax()][0]],  # BR
#             [points[d.argmax()][1], points[d.argmax()][0]],  # BL
#         ])
#         return self.corners


# # Warp

# def warp_to_corners(gray: np.ndarray,
#                     corners: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
#     """
#     Perspective-warp gray to a rectangle whose size matches the aspect
#     ratio of the detected corners (not forced square).

#     Returns (warped_gray, M) or None if corners is None.
#     """
#     if corners is None:
#         return None

#     tl, tr, br, bl = corners

#     # Width = average of top and bottom edge lengths
#     w = int((np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2)
#     # Height = average of left and right edge lengths
#     h = int((np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2)

#     if w < 10 or h < 10:
#         return None

#     dst = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
#     M   = cv2.getPerspectiveTransform(corners, dst)
#     return cv2.warpPerspective(gray, M, (w, h)), M


# # Step/difference based detection
# class StepDiff:
#     """
#     Edge detector using stepsize finite difference -> binary -> scan lines.

#     Algorithm
#     ---------
#     1. Build diff array:
#          diff_x[y, x] = |gray[y, x+step] - gray[y, x-step]|   (horizontal)
#          diff_y[y, x] = |gray[y+step, x] - gray[y-step, x]|   (vertical)
#          diff[y, x]   = max(diff_x, diff_y) > threshold  -> 0 or 1

#     2. Scan every row AND every column of the binary diff array.
#        In each 1-D scan:
#          - A run of 1s = one detected edge
#          - outer  = first 1  (rising edge)
#          - inner  = last  1  (falling edge)
#          - center = (outer + inner) // 2

#     Parameters
#     ----------
#     gray      : grayscale uint8 image
#     threshold : minimum difference to count as an edge (default 30)
#     step      : finite-difference step size in pixels  (default 2)
#     """

#     def __init__(self, gray: np.ndarray, threshold: int = 30, step: int = 2):
#         self.gray      = gray
#         self.threshold = threshold
#         self.step      = step

#         # Results — populated by detect()
#         self.diff_arr = None
#         self.ys       = np.array([], dtype=np.int32)
#         self.xs       = np.array([], dtype=np.int32)   # column-scan row positions
#         self.outers   = np.array([], dtype=np.int32)
#         self.inners   = np.array([], dtype=np.int32)
#         self.centers  = np.array([], dtype=np.int32)
#         self.widths   = np.array([], dtype=np.int32)

#     def detect(self) -> 'StepDiff':
#         self._build_diff()

#         h, w = self.diff_arr.shape

#         row_ys      = []
#         row_outers  = []
#         row_inners  = []

#         col_ys      = []
#         col_outers  = []
#         col_inners  = []

#         # Scan rows
#         for y in range(h):
#             result = self._scan_line(self.diff_arr[y, :])
#             if result is not None:
#                 o, i = result
#                 row_ys.append(np.full(len(o), y, dtype=np.int32))
#                 row_outers.append(o)
#                 row_inners.append(i)

#         # Scan columns
#         for x in range(w):
#             result = self._scan_line(self.diff_arr[:, x])
#             if result is not None:
#                 o, i = result
#                 col_ys.append(o)          # o = row positions for column scan
#                 col_outers.append(np.full(len(o), x, dtype=np.int32))
#                 col_inners.append(np.full(len(o), x, dtype=np.int32))

#         # Combine rows
#         if row_ys:
#             all_ys     = np.concatenate(row_ys)
#             all_outers = np.concatenate(row_outers)
#             all_inners = np.concatenate(row_inners)
#         else:
#             all_ys = all_outers = all_inners = np.array([], dtype=np.int32)

#         self.ys      = all_ys
#         self.outers  = all_outers
#         self.inners  = all_inners
#         self.centers = (all_outers + all_inners) // 2
#         self.widths  = all_inners - all_outers

#         return self

#     def draw(self, circle_r: int = 3) -> np.ndarray:
#         out = cv2.cvtColor(self.gray, cv2.COLOR_GRAY2BGR)
#         if len(self.ys) > 0:
#             for y, c in zip(self.ys, self.centers):
#                 cv2.circle(out, (int(c), int(y)), circle_r, (0, 0, 255), -1)
#         return out

#     def draw_corners(self, out: np.ndarray,
#                      corners: np.ndarray | None) -> np.ndarray:
#         if corners is None:
#             return out
#         colors = [(255, 255, 255)] * 4   # all white
#         for i, (cx, cy) in enumerate(corners):
#             cv2.circle(out, (int(cx), int(cy)), 8, colors[i], -1)
#             cv2.circle(out, (int(cx), int(cy)), 8, (0, 0, 0),   2)
#         return out

#     def _build_diff(self):
#         s = self.step
#         g = self.gray.astype(np.int16)

#         # Horizontal diff
#         diff_x = np.abs(g[:, s:] - g[:, :-s])
#         diff_x = np.pad(diff_x, ((0, 0), (s // 2, s - s // 2)), mode='constant')

#         # Vertical diff
#         diff_y = np.abs(g[s:, :] - g[:-s, :])
#         diff_y = np.pad(diff_y, ((s // 2, s - s // 2), (0, 0)), mode='constant')

#         # Align shapes after padding
#         h = min(diff_x.shape[0], diff_y.shape[0])
#         w = min(diff_x.shape[1], diff_y.shape[1])
#         diff_x = diff_x[:h, :w]
#         diff_y = diff_y[:h, :w]

#         self.diff_arr = (
#             np.maximum(diff_x, diff_y) > self.threshold
#         ).astype(np.uint8)

#     @staticmethod
#     def _scan_line(line: np.ndarray):
#         """
#         Find all runs of 1s in a binary 1-D array.
#         Returns (outers, inners) as int32 arrays, or None if no runs.
#         outer = index of first 1, inner = index of last 1 in each run.
#         """
#         d      = np.diff(line.astype(np.int8))
#         starts = np.where(d ==  1)[0] + 1   # +1: diff index is one behind
#         ends   = np.where(d == -1)[0]

#         # Handle edge cases: line starts or ends inside a run
#         if len(line) > 0 and line[0] == 1:
#             starts = np.concatenate([[0], starts])
#         if len(line) > 0 and line[-1] == 1:
#             ends = np.concatenate([ends, [len(line) - 1]])

#         n = min(len(starts), len(ends))
#         if n == 0:
#             return None

#         # Keep only valid pairs
#         mask   = starts[:n] < ends[:n]
#         starts = starts[:n][mask]
#         ends   = ends[:n][mask]

#         if len(starts) == 0:
#             return None

#         return starts.astype(np.int32), ends.astype(np.int32)


# # Derivative based detection
# class Derivative:
#     """
#     Edge detector using 1st and 2nd derivative zero-crossing method.

#     Algorithm  (per scan-line profile)
#     -----------------------------------
#     1. Smooth profile with 3-tap Gaussian [0.25, 0.5, 0.25]
#     2. d1 = np.diff(profile)        — 1st derivative
#        d2 = np.diff(d1)             — 2nd derivative
#     3. Rising edge top  = first d1 sample above threshold
#                           (refined by d2 +→− zero crossing if found)
#     4. Falling edge top = last  d1 sample above threshold
#                           (refined by d2 −→+ zero crossing if found)
#     5. Center           = midpoint of rising/falling
#                           (refined by parabolic peak of d1 if inside band)

#     Scans both rows and columns for better coverage of skewed lines.

#     Parameters
#     ----------
#     gray      : grayscale uint8 image
#     threshold : minimum d1 value to count as edge  (default 15)
#     step      : finite-difference step for gradient (default 2)
#     blur      : Gaussian blur kernel before scanning (0 = off, must be odd)
#     """

#     def __init__(self,
#                  gray:      np.ndarray,
#                  threshold: int = 15,
#                  step:      int = 2,
#                  blur:      int = 3):
#         self.gray      = gray
#         self.threshold = threshold
#         self.step      = step
#         self.blur      = blur if blur % 2 == 1 else blur + 1

#         self.ys      = np.array([], dtype=np.int32)
#         self.outers  = np.array([], dtype=np.int32)
#         self.inners  = np.array([], dtype=np.int32)
#         self.centers = np.array([], dtype=np.int32)
#         self.widths  = np.array([], dtype=np.int32)

#     def detect(self) -> 'Derivative':
#         src = (cv2.GaussianBlur(self.gray, (self.blur, self.blur), 0)
#                if self.blur > 1 else self.gray)

#         h, w = src.shape
#         row_ys = [];      row_outers = [];  row_inners = []
#         col_ys = [];      col_outers = [];  col_inners = []

#         # Scan rows
#         for y in range(h):
#             result = self._scan_profile(src[y, :].astype(np.float32))
#             if result is not None:
#                 o, i = result
#                 row_ys.append(np.full(len(o), y, dtype=np.int32))
#                 row_outers.append(o)
#                 row_inners.append(i)

#         # Scan columns
#         for x in range(w):
#             result = self._scan_profile(src[:, x].astype(np.float32))
#             if result is not None:
#                 o, i = result
#                 col_ys.append(o)
#                 col_outers.append(np.full(len(o), x, dtype=np.int32))
#                 col_inners.append(np.full(len(o), x, dtype=np.int32))

#         if row_ys:
#             all_ys     = np.concatenate(row_ys)
#             all_outers = np.concatenate(row_outers)
#             all_inners = np.concatenate(row_inners)
#         else:
#             all_ys = all_outers = all_inners = np.array([], dtype=np.int32)

#         self.ys      = all_ys
#         self.outers  = all_outers
#         self.inners  = all_inners
#         self.centers = (all_outers + all_inners) // 2
#         self.widths  = all_inners - all_outers

#         return self

#     def draw(self, circle_r: int = 3) -> np.ndarray:
#         out = cv2.cvtColor(self.gray, cv2.COLOR_GRAY2BGR)
#         if len(self.ys) > 0:
#             for y, c in zip(self.ys, self.centers):
#                 cv2.circle(out, (int(c), int(y)), circle_r, (0, 0, 255), -1)
#         return out

#     def draw_corners(self, out: np.ndarray,
#                      corners: np.ndarray | None) -> np.ndarray:
#         if corners is None:
#             return out
#         for cx, cy in corners:
#             cv2.circle(out, (int(cx), int(cy)), 10, (255, 255, 255), -1)
#             cv2.circle(out, (int(cx), int(cy)), 10, (0,   0,   0),    2)
#         return out

#     def _scan_profile(self, profile: np.ndarray):
#         """
#         Find all edges in a 1-D intensity profile via d1/d2 method.
#         Returns (outers, inners) as int32 arrays, or None.
#         """
#         smooth_k = np.array([0.25, 0.50, 0.25], dtype=np.float32)
#         p   = np.convolve(profile, smooth_k, mode='same')
#         d1  = np.diff(p)
#         nd1 = len(d1)
#         margin = 2

#         if nd1 <= 2 * margin:
#             return None

#         d2 = np.diff(d1)

#         outers_list = []
#         inners_list = []

#         # Walk through d1 looking for above-threshold bands
#         i = margin
#         while i < nd1 - margin:
#             if d1[i] < self.threshold:
#                 i += 1
#                 continue

#             # Found start of a band — find the full extent
#             band_start = i
#             while i < nd1 - margin and d1[i] >= self.threshold:
#                 i += 1
#             band_end = i - 1

#             # Rising edge top: d2 +→− left of band_start
#             outer_f = self._refine_zc(d2, band_start, pos_to_neg=True,
#                                       search_range=4, direction='left')
#             outer = int(np.clip(round(outer_f + 0.5), 0, len(profile) - 1))

#             # Falling edge top: d2 −→+ right of band_end
#             inner_f = self._refine_zc(d2, band_end, pos_to_neg=False,
#                                       search_range=4, direction='right')
#             inner = int(np.clip(round(inner_f + 0.5), 0, len(profile) - 1))

#             # Fallback: if refinement collapsed outer >= inner, use raw band
#             if inner <= outer:
#                 outer = int(np.clip(band_start, 0, len(profile) - 1))
#                 inner = int(np.clip(band_end + 1, 0, len(profile) - 1))

#             if inner > outer:
#                 outers_list.append(outer)
#                 inners_list.append(inner)

#         if not outers_list:
#             return None

#         return (np.array(outers_list, dtype=np.int32),
#                 np.array(inners_list, dtype=np.int32))

#     @staticmethod
#     def _refine_zc(d2: np.ndarray, near: int,
#                    pos_to_neg: bool, search_range: int,
#                    direction: str = 'left') -> float:
#         """
#         Find nearest d2 zero crossing of given polarity near `near`.
#         direction='left'  — search leftward  (for rising edge top)
#         direction='right' — search rightward (for falling edge top)
#         Returns subpixel d2 index, or float(near) if none found.
#         """
#         if direction == 'left':
#             lo = max(0, near - search_range)
#             hi = min(len(d2) - 1, near + 1)
#             indices = range(lo, hi)
#         else:
#             lo = max(0, near - 1)
#             hi = min(len(d2) - 1, near + search_range)
#             indices = range(lo, hi)

#         for j in indices:
#             a, b = d2[j], d2[j + 1]
#             if a == b:
#                 continue
#             if (a > 0) != (b > 0) and ((a > 0) == pos_to_neg):
#                 t = -a / (b - a)
#                 return j + t
#         return float(near)

#     @staticmethod
#     def _parabolic(sig: np.ndarray, i: int) -> float:
#         if i <= 0 or i >= len(sig) - 1:
#             return float(i)
#         a, b, c = sig[i - 1], sig[i], sig[i + 1]
#         denom = 2.0 * (2.0 * b - a - c)
#         if abs(denom) < 1e-6:
#             return float(i)
#         return float(i) + (a - c) / denom

import cv2
import numpy as np


# ── Base ──────────────────────────────────────────────────────────────────────

class EdgeDetector:
    """Common interface and helpers for all edge detectors."""

    gray:          np.ndarray
    diffArr:       np.ndarray       # binary mask (StepDiff) or blurred src (Derivative)
    ys:            np.ndarray       # row of each detected edge centre
    centers:       np.ndarray       # column of each detected edge centre
    slots:         int
    slotThreshold: int              # percent (0-100)

    def draw(self, r: int = 3) -> np.ndarray:
        out = cv2.cvtColor(self.gray, cv2.COLOR_GRAY2BGR)
        for y, c in zip(self.ys, self.centers):
            cv2.circle(out, (int(c), int(y)), r, (0, 0, 255), -1)
        return out

    def _store(self, outers: np.ndarray, inners: np.ndarray, ys: np.ndarray) -> None:
        """Finalise ys / centers / widths from raw outers+inners arrays."""
        self.ys      = ys
        self.centers = (outers + inners) // 2
        self.widths  = inners - outers


# ── Step-difference detector ──────────────────────────────────────────────────

class StepDiff(EdgeDetector):
    """
    Binary edge map via finite difference, then scan each row for runs of 1s.

    diffArr[y,x] = max(|gray[y, x±step]|, |gray[y±step, x]|) > threshold
    """

    def __init__(self, gray: np.ndarray,
                 diffThreshold: int = 30, step: int = 2,
                 slots: int = 4,         slotThreshold: int = 20) -> None:
        self.gray, self.threshold, self.step = gray, diffThreshold, step
        self.slots, self.slotThreshold       = slots, slotThreshold
        self.ys = self.centers = self.widths = np.array([], dtype=np.int32)
        self._detect()

    def _detect(self) -> None:
        s, g = self.step, self.gray.astype(np.int16)
        dx = np.pad(np.abs(g[:, s:] - g[:, :-s]), ((0,0),(s//2, s-s//2)), mode="edge")
        dy = np.pad(np.abs(g[s:, :] - g[:-s, :]), ((s//2, s-s//2),(0,0)), mode="edge")
        self.diffArr = (np.maximum(dx, dy) > self.threshold).astype(np.uint8)

        # Vectorized run-length encoding across all rows at once
        # Pad a zero column on each side to catch runs at boundaries
        padded  = np.pad(self.diffArr, ((0,0),(1,1)), mode="constant")
        diff    = np.diff(padded.astype(np.int8), axis=1)

        starts_r, starts_c = np.where(diff ==  1)   # rising edges
        ends_r,   ends_c   = np.where(diff == -1)   # falling edges

        # Match starts to ends within the same row
        order    = np.argsort(starts_r * padded.shape[1] + starts_c)
        starts_r = starts_r[order];  starts_c = starts_c[order]
        order    = np.argsort(ends_r   * padded.shape[1] + ends_c)
        ends_r   = ends_r[order];    ends_c   = ends_c[order]

        n        = min(len(starts_r), len(ends_r))
        mask     = (starts_r[:n] == ends_r[:n]) & (starts_c[:n] < ends_c[:n])

        self.ys      = starts_r[:n][mask].astype(np.int32)
        self.outers  = starts_c[:n][mask].astype(np.int32)
        self.inners  = ends_c[:n][mask].astype(np.int32)
        self.centers = ((self.outers + self.inners) // 2).astype(np.int32)
        self.widths  = (self.inners - self.outers).astype(np.int32)
    
    # def _detect(self) -> None:
    #     s, g = self.step, self.gray.astype(np.int16)

    #     dx = np.pad(np.abs(g[:, s:] - g[:, :-s]),
    #                 ((0,0), (s//2, s - s//2)), mode="edge")
    #     dy = np.pad(np.abs(g[s:, :] - g[:-s, :]),
    #                 ((s//2, s - s//2), (0,0)), mode="edge")

    #     self.diffArr = (np.maximum(dx, dy) > self.threshold).astype(np.uint8)

    #     row_ys, outers, inners = [], [], []

    #     for y, row in enumerate(self.diffArr):
    #         result = self._runs(row)
    #         if result is not None:
    #             o, i = result
    #             if len(o) >= 2:
    #                 row_ys.append(np.full(len(o) - 1, y, dtype=np.int32))
    #                 outers.append(i[:-1]);  inners.append(o[1:])
    #             else:
    #                 row_ys.append(np.full(len(o), y, dtype=np.int32))
    #                 outers.append(o);       inners.append(i)

    #     for x in range(self.diffArr.shape[1]):
    #         result = self._runs(self.diffArr[:, x])
    #         if result is not None:
    #             o, i = result
    #             if len(o) >= 2:
    #                 row_ys.append((i[:-1] + o[1:]) // 2)
    #                 outers.append(np.full(len(o) - 1, x, dtype=np.int32))
    #                 inners.append(np.full(len(o) - 1, x, dtype=np.int32))
    #             else:
    #                 row_ys.append((o + i) // 2)
    #                 outers.append(np.full(len(o), x, dtype=np.int32))
    #                 inners.append(np.full(len(o), x, dtype=np.int32))

    #     if row_ys:
    #         self._store(np.concatenate(outers), np.concatenate(inners),
    #                     np.concatenate(row_ys))
    #     else:
    #         self.ys = self.centers = self.widths = np.array([], dtype=np.int32)
    
    # def _detect(self) -> None:
    #     s, g = self.step, self.gray.astype(np.int16)

    #     dx = np.pad(np.abs(g[:, s:] - g[:, :-s]),
    #                 ((0,0), (s//2, s - s//2)), mode="edge")
    #     dy = np.pad(np.abs(g[s:, :] - g[:-s, :]),
    #                 ((s//2, s - s//2), (0,0)), mode="edge")

    #     self.diffArr = (np.maximum(dx, dy) > self.threshold).astype(np.uint8)

    #     row_ys, outers, inners = [], [], []
    #     for y, row in enumerate(self.diffArr):
    #         result = self._runs(row)
    #         if result is not None:
    #             o, i = result
    #             row_ys.append(np.full(len(o), y, dtype=np.int32))
    #             outers.append(o);  inners.append(i)

    #     if row_ys:
    #         self._store(np.concatenate(outers), np.concatenate(inners),
    #                     np.concatenate(row_ys))
    #     else:
    #         self.ys = self.centers = self.widths = np.array([], dtype=np.int32)

    @staticmethod
    def _runs(line: np.ndarray):
        """All runs of 1s in a binary 1-D array → (starts, ends) or None."""
        d      = np.diff(line.astype(np.int8))
        starts = np.where(d ==  1)[0] + 1
        ends   = np.where(d == -1)[0]
        if line[0]  == 1: starts = np.concatenate([[0], starts])
        if line[-1] == 1: ends   = np.concatenate([ends, [len(line) - 1]])
        n = min(len(starts), len(ends))
        if n == 0: return None
        mask = starts[:n] < ends[:n]
        s, e = starts[:n][mask], ends[:n][mask]
        return (s.astype(np.int32), e.astype(np.int32)) if len(s) else None


# ── Derivative detector ───────────────────────────────────────────────────────

class Derivative(EdgeDetector):
    """
    Edge detector via 1st/2nd derivative zero-crossings on each row profile.

    Per row: smooth → d1 → threshold bands → d2 zero-crossing refinement.
    """

    def __init__(self, gray: np.ndarray,
                 threshold: int = 15, step: int = 2, blur: int = 3,
                 slots: int = 4,      slotThreshold: int = 20) -> None:
        self.gray, self.threshold  = gray, threshold
        self.blur                  = blur if blur % 2 else blur + 1
        self.slots, self.slotThreshold = slots, slotThreshold
        self.ys = self.centers = self.widths = np.array([], dtype=np.int32)
        self._detect()
    
    # Vectorized but no canny
    def _detect(self) -> None:
        src = cv2.GaussianBlur(self.gray, (self.blur, self.blur), 0) if self.blur > 1 else self.gray
        self.diffArr = src

        # Smooth + d1 across all rows at once — no Python loop
        k   = np.array([0.25, 0.5, 0.25], dtype=np.float32)
        src_f = src.astype(np.float32)

        # Convolve each row: apply kernel via 3-tap weighted sum (no scipy needed)
        smooth = (np.roll(src_f, 1, axis=1) * 0.25
                + src_f                     * 0.50
                + np.roll(src_f,-1, axis=1) * 0.25)

        d1 = np.diff(smooth, axis=1)                       # (H, W-1)

        # Binary mask: where d1 exceeds threshold
        above = (d1 >= self.threshold)                     # (H, W-1)

        # Vectorized run detection — same pad/diff trick as StepDiff
        padded = np.pad(above.astype(np.int8), ((0,0),(1,1)), mode="constant")
        delta  = np.diff(padded, axis=1)

        starts_r, starts_c = np.where(delta ==  1)
        ends_r,   ends_c   = np.where(delta == -1)

        order    = np.argsort(starts_r * padded.shape[1] + starts_c)
        starts_r = starts_r[order];  starts_c = starts_c[order]
        order    = np.argsort(ends_r   * padded.shape[1] + ends_c)
        ends_r   = ends_r[order];    ends_c   = ends_c[order]

        n    = min(len(starts_r), len(ends_r))
        mask = (starts_r[:n] == ends_r[:n]) & (starts_c[:n] < ends_c[:n])

        self.ys      = starts_r[:n][mask].astype(np.int32)
        self.outers  = starts_c[:n][mask].astype(np.int32)
        self.inners  = ends_c[:n][mask].astype(np.int32)
        self.centers = ((self.outers + self.inners) // 2).astype(np.int32)
        self.widths  = (self.inners - self.outers).astype(np.int32)
        
    # Using canny 
    # def _detect(self) -> None:
    #     src = cv2.GaussianBlur(self.gray, (self.blur, self.blur), 0) if self.blur > 1 else self.gray
    #     self.diffArr = src

    #         # Using canny
    #     canny = cv2.Canny(src, self.threshold * 3, self.threshold * 6)

    #     ys, xs = np.where(canny > 0)
    #     self.ys      = ys.astype(np.int32)
    #     self.centers = xs.astype(np.int32)
    #     self.outers  = self.centers
    #     self.inners  = self.centers
    #     self.widths  = np.zeros_like(self.centers)
    
    # Python loop, works but slow
    # def _detect(self) -> None:
    #     src = cv2.GaussianBlur(self.gray, (self.blur, self.blur), 0) if self.blur > 1 else self.gray
    #     self.diffArr = src   # shape used by corner finders

    #     row_ys, outers, inners = [], [], []
    #     for y in range(self.diffArr.shape[0]):
    #         result = self._scan(self.diffArr[y,:])
    #         if result is not None:
    #             o, i = result
    #             row_ys.append(np.full(len(o), y, dtype=np.int32))
    #             outers.append(o);  inners.append(i)
                
    #     for x in range(self.diffArr.shape[1]):
    #         result = self._scan(self.diffArr[:, x])
    #         if result is not None:
    #             oo, ii = result
    #             row_ys.append(oo)
    #             outers.append(np.full(len(oo), x, dtype=np.int32))
    #             inners.append(np.full(len(oo), x, dtype=np.int32))

    #     if row_ys:
    #         self._store(np.concatenate(outers), np.concatenate(inners),
    #                     np.concatenate(row_ys))
    #     else:
    #         self.ys = self.centers = self.widths = np.array([], dtype=np.int32)
            
    def _scan(self, profile: np.ndarray):
        p  = np.convolve(profile, [0.25, 0.5, 0.25], mode="same")
        d1 = np.diff(p);   d2 = np.diff(d1)
        if len(d1) <= 4: return None

        os, ins, i = [], [], 2
        while i < len(d1) - 2:
            if d1[i] < self.threshold:
                i += 1;  continue
            b0 = i
            while i < len(d1) - 2 and d1[i] >= self.threshold: i += 1
            b1 = i - 1

            o = int(np.clip(round(self._zc(d2, b0, True,  "left")  + 0.5), 0, len(profile)-1))
            n = int(np.clip(round(self._zc(d2, b1, False, "right") + 0.5), 0, len(profile)-1))
            if n <= o:
                o, n = int(np.clip(b0,   0, len(profile)-1)), \
                       int(np.clip(b1+1, 0, len(profile)-1))
            if n > o:
                os.append(o);  ins.append(n)

        return (np.array(os, dtype=np.int32), np.array(ins, dtype=np.int32)) if os else None

    @staticmethod
    def _zc(d2: np.ndarray, near: int, pos_to_neg: bool, direction: str) -> float:
        """Subpixel d2 zero-crossing of given polarity closest to *near*."""
        lo, hi = ((max(0, near-4), min(len(d2)-1, near+1)) if direction == "left"
                  else (max(0, near-1), min(len(d2)-1, near+4)))
        for j in range(lo, hi):
            a, b = d2[j], d2[j+1]
            if a != b and (a > 0) != (b > 0) and (a > 0) == pos_to_neg:
                return j + (-a / (b - a))
        return float(near)
    
    
    class PreciseEdge(EdgeDetector):
        """
        Pyramid → Sobel → NMS → hysteresis → subpixel centers.
        Single class, replaces both StepDiff and Derivative.
        """

    def __init__(self, gray: np.ndarray,
                 threshold:     int   = 30,
                 slots:         int   = 4,
                 slotThreshold: int   = 20,
                 subpixel:      bool  = True,
                 pyramid:       bool  = True) -> None:
        self.gray, self.threshold      = gray, threshold
        self.slots, self.slotThreshold = slots, slotThreshold
        self.subpixel, self.pyramid    = subpixel, pyramid
        self.centers_f                 = np.array([], dtype=np.float32)
        self.ys = self.centers = self.widths = np.array([], dtype=np.int32)
        self._detect()

    def _detect(self) -> None:
        g = self.gray

        # ── Pyramid ROI mask ──────────────────────────────────────────
        if self.pyramid:
            small  = cv2.resize(g, (g.shape[1]//2, g.shape[0]//2),
                                interpolation=cv2.INTER_AREA)
            gx_s   = cv2.Sobel(small, cv2.CV_16S, 1, 0, ksize=3)
            gy_s   = cv2.Sobel(small, cv2.CV_16S, 0, 1, ksize=3)
            mask_s = (np.maximum(np.abs(gx_s), np.abs(gy_s)) > self.threshold).astype(np.uint8)
            mask_s = cv2.dilate(mask_s, np.ones((5,5), np.uint8))
            roi    = cv2.resize(mask_s, (g.shape[1], g.shape[0]),
                                interpolation=cv2.INTER_NEAREST).astype(bool)
        else:
            roi    = np.ones(g.shape, dtype=bool)

        # ── Full-res Sobel ────────────────────────────────────────────
        gx  = cv2.Sobel(g, cv2.CV_16S, 1, 0, ksize=3)
        gy  = cv2.Sobel(g, cv2.CV_16S, 0, 1, ksize=3)
        mag = np.maximum(np.abs(gx), np.abs(gy)).astype(np.float32)
        mag[~roi] = 0

        # ── NMS ───────────────────────────────────────────────────────
        thin = self._nms(mag, gx, gy)

        # ── Hysteresis ────────────────────────────────────────────────
        self.diffArr = self._hysteresis(thin, mag)

        # ── Vectorised run extraction ─────────────────────────────────
        self._extract_runs(mag)

    def _nms(self, mag: np.ndarray,
             gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
        angle = np.arctan2(gy.astype(np.float32),
                           gx.astype(np.float32)) * 180 / np.pi % 180
        q     = ((angle + 22.5) // 45 % 4).astype(np.uint8)
        out   = mag.copy()

        for d, (dy, dx) in enumerate([(0,1),(1,1),(1,0),(1,-1)]):
            mask = q == d
            fwd  = np.roll(np.roll(mag, -dy, 0), -dx, 1)
            bwd  = np.roll(np.roll(mag,  dy, 0),  dx, 1)
            out[mask & ((out < fwd) | (out < bwd))] = 0

        return (out > self.threshold).astype(np.uint8)

    def _hysteresis(self, thin: np.ndarray, mag: np.ndarray) -> np.ndarray:
        weak       = (mag > self.threshold * 0.4).astype(np.uint8) & thin
        strong     = thin.copy()
        _, labels  = cv2.connectedComponents(weak)
        strong_ids = np.unique(labels[strong > 0])
        out        = np.zeros_like(thin)
        for sid in strong_ids[strong_ids != 0]:
            out[labels == sid] = 1
        return out

    def _extract_runs(self, mag: np.ndarray) -> None:
        padded   = np.pad(self.diffArr, ((0,0),(1,1)), mode="constant")
        delta    = np.diff(padded.astype(np.int8), axis=1)
        ys_s, xs_s = np.where(delta ==  1)
        ys_e, xs_e = np.where(delta == -1)

        w        = padded.shape[1]
        idx_s    = np.argsort(ys_s * w + xs_s)
        idx_e    = np.argsort(ys_e * w + xs_e)
        ys_s, xs_s = ys_s[idx_s], xs_s[idx_s]
        ys_e, xs_e = ys_e[idx_e], xs_e[idx_e]

        n     = min(len(ys_s), len(ys_e))
        valid = (ys_s[:n] == ys_e[:n]) & (xs_s[:n] < xs_e[:n])

        self.ys     = ys_s[:n][valid].astype(np.int32)
        self.outers = xs_s[:n][valid].astype(np.int32)
        self.inners = xs_e[:n][valid].astype(np.int32)
        self.widths = (self.inners - self.outers).astype(np.int32)

        if self.subpixel:
            self._subpixel_centers(mag)
        else:
            self.centers   = ((self.outers + self.inners) >> 1).astype(np.int32)
            self.centers_f = self.centers.astype(np.float32)

    def _subpixel_centers(self, mag: np.ndarray) -> None:
        centers_f = ((self.outers + self.inners) / 2).astype(np.float32)
        for k in range(len(self.outers)):
            y, o, i  = self.ys[k], self.outers[k], self.inners[k]
            profile  = mag[y, o:i+1]
            peak     = int(np.argmax(profile))
            if 0 < peak < len(profile) - 1:
                a, b, c  = profile[peak-1], profile[peak], profile[peak+1]
                denom    = 2*(2*b - a - c)
                centers_f[k] = o + peak + ((a-c)/denom if abs(denom) > 1e-6 else 0)
        self.centers_f = centers_f
        self.centers   = centers_f.astype(np.int32)
        
    # # HoughLinesP
    # def fit_lines(diffArr: np.ndarray,
    #           min_length: int = 30,
    #           max_gap:    int = 5) -> np.ndarray:
    #     """
    #     Returns array of (x1,y1,x2,y2) line segments.
    #     min_length: discard segments shorter than this
    #     max_gap:    bridge gaps in edge chain up to this many pixels
    #     """
    #     return cv2.HoughLinesP(
    #         diffArr,
    #         rho=1, theta=np.pi/180,
    #         threshold=40,
    #         minLineLength=min_length,
    #         maxLineGap=max_gap
    #     )
    
    # Ransac
    # def _ransac_line(ys: np.ndarray, xs: np.ndarray,
    #              iterations: int = 100,
    #              inlier_dist: float = 1.5) -> tuple | None:
    #     """
    #     Fit a line to (xs, ys) via RANSAC.
    #     Returns (vx, vy, x0, y0) — direction vector + point on line, or None.
    #     """
    #     best_inliers = 0
    #     best_line    = None
    #     n            = len(ys)
    #     if n < 2: return None

    #     pts = np.column_stack([xs, ys]).astype(np.float32)

    #     for _ in range(iterations):
    #         i, j  = np.random.choice(n, 2, replace=False)
    #         p1, p2 = pts[i], pts[j]
    #         d     = p2 - p1
    #         norm  = np.linalg.norm(d)
    #         if norm < 1e-6: continue
    #         d /= norm

    #         # Distance from all points to this line
    #         diff    = pts - p1
    #         dist    = np.abs(diff[:,0]*d[1] - diff[:,1]*d[0])
    #         inliers = (dist < inlier_dist).sum()

    #         if inliers > best_inliers:
    #             best_inliers = inliers
    #             best_line    = (d[0], d[1], p1[0], p1[1])

    #     return best_line
