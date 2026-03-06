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
    
    def _detect(self) -> None:
        s, g = self.step, self.gray.astype(np.int16)

        dx = np.pad(np.abs(g[:, s:] - g[:, :-s]),
                    ((0,0), (s//2, s - s//2)), mode="edge")
        dy = np.pad(np.abs(g[s:, :] - g[:-s, :]),
                    ((s//2, s - s//2), (0,0)), mode="edge")

        self.diffArr = (np.maximum(dx, dy) > self.threshold).astype(np.uint8)

        row_ys, outers, inners = [], [], []
        for y, row in enumerate(self.diffArr):
            result = self._runs(row)
            if result is not None:
                o, i = result
                row_ys.append(np.full(len(o), y, dtype=np.int32))
                outers.append(o);  inners.append(i)

        if row_ys:
            self._store(np.concatenate(outers), np.concatenate(inners),
                        np.concatenate(row_ys))
        else:
            self.ys = self.centers = self.widths = np.array([], dtype=np.int32)

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

    # def _detect(self) -> None:
    #     src = cv2.GaussianBlur(self.gray, (self.blur, self.blur), 0) if self.blur > 1 else self.gray
    #     self.diffArr = src

    #     row_ys, outers, inners = [], [], []

    #     for y in range(src.shape[0]):
    #         result = self._scan(src[y].astype(np.float32))
    #         if result is not None:
    #             o, i = result
    #             if len(o) >= 2:
    #                 row_ys.append(np.full(len(o) - 1, y, dtype=np.int32))
    #                 outers.append(i[:-1]);  inners.append(o[1:])
    #             else:
    #                 row_ys.append(np.full(len(o), y, dtype=np.int32))
    #                 outers.append(o);       inners.append(i)

    #     for x in range(src.shape[1]):
    #         result = self._scan(src[:, x].astype(np.float32))
    #         if result is not None:
    #             o, i = result
    #             if len(o) >= 2:
    #                 row_ys.append((i[:-1] + o[1:]) // 2)           # gap centre rows
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
    
    def _detect(self) -> None:
        src = cv2.GaussianBlur(self.gray, (self.blur, self.blur), 0) if self.blur > 1 else self.gray
        self.diffArr = src   # shape used by corner finders

        row_ys, outers, inners = [], [], []
        for y in range(src.shape[0]):
            result = self._scan(self.diffArr[y,:])
            if result is not None:
                o, i = result
                row_ys.append(np.full(len(o), y, dtype=np.int32))
                outers.append(o);  inners.append(i)
                
        for x in range(self.diffArr.shape[1]):
            result = self._scan(self.diffArr[:, x])
            if result is not None:
                oo, ii = result
                row_ys.append(oo)
                outers.append(np.full(len(oo), x, dtype=np.int32))
                inners.append(np.full(len(oo), x, dtype=np.int32))

        if row_ys:
            self._store(np.concatenate(outers), np.concatenate(inners),
                        np.concatenate(row_ys))
        else:
            self.ys = self.centers = self.widths = np.array([], dtype=np.int32)
            
            

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
