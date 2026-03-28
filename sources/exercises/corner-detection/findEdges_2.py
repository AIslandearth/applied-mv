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
