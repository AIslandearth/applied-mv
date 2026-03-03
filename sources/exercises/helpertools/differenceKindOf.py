# img  = cv2.imread("sudoku.png")
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# diff = Diff(gray, threshold=30, step=2)
# diff.detect()

# # Original with detected edges
# original = diff.draw()

# # Reoriented — perspective corrected
# warped = diff.warp()

# if warped is not None:
#     # Run detection again on warped image
#     diff_warped = Diff(warped, threshold=30, step=2)
#     diff_warped.detect()
#     result = diff_warped.draw()

#     cv2.imshow("Original",    original)
#     cv2.imshow("Reoriented",  result)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

import cv2
import numpy as np

class Diff:

    def __init__(self, gray, threshold=30, step=2):
        self.gray      = gray
        self.threshold = threshold
        self.step      = step
        self.diff_arr  = None
        self.ys        = np.array([], dtype=int)
        self.outers    = np.array([], dtype=int)
        self.inners    = np.array([], dtype=int)
        self.centers   = np.array([], dtype=int)
        self.widths    = np.array([], dtype=int)
        self.corners   = None


    def _build_diff(self):
        diff_x = np.abs(
            self.gray[:, self.step:].astype(np.int16) -
            self.gray[:, :-self.step].astype(np.int16)
        )
        diff_y = np.abs(
            self.gray[self.step:, :].astype(np.int16) -
            self.gray[:-self.step, :].astype(np.int16)
        )
        diff_x        = np.pad(diff_x, ((0,0),(0,self.step)), mode='constant')
        diff_y        = np.pad(diff_y, ((0,self.step),(0,0)), mode='constant')
        self.diff_arr = (np.maximum(diff_x, diff_y) > self.threshold).astype(np.uint8)


    def _scan_line(self, line):
        diffs  = np.diff(line.astype(np.int8))
        starts = np.where(diffs ==  1)[0]
        ends   = np.where(diffs == -1)[0]
        n      = min(len(starts), len(ends))
        if n == 0:
            return None
        outers = starts[:n]
        inners = ends[:n]
        return outers, inners, (outers+inners)//2, inners-outers


    def _find_corners(self):
        if len(self.ys) == 0:
            return None
        points = np.column_stack([self.ys, self.centers])
        s      = points[:, 0] + points[:, 1]
        d      = points[:, 0] - points[:, 1]
        return np.float32([
            [points[s.argmin()][1], points[s.argmin()][0]],
            [points[d.argmin()][1], points[d.argmin()][0]],
            [points[s.argmax()][1], points[s.argmax()][0]],
            [points[d.argmax()][1], points[d.argmax()][0]]
        ])


    def detect(self):
        self._build_diff()

        h          = self.diff_arr.shape[0]
        all_ys     = []
        all_outers = []
        all_inners = []

        # Scan all rows
        y = 0
        while y < h:
            result = self._scan_line(self.diff_arr[y, :])
            if result is not None:
                o, i, _, _ = result
                all_ys.extend([y]  * len(o))
                all_outers.extend(o)
                all_inners.extend(i)
            y += 1

        self.ys      = np.array(all_ys)
        self.outers  = np.array(all_outers)
        self.inners  = np.array(all_inners)
        self.centers = (self.outers + self.inners) // 2
        self.widths  = self.inners - self.outers
        self.corners = self._find_corners()

        return self


    def warp(self, size=500):
        if self.corners is None:
            return None
        dst = np.float32([[0,0],[size,0],[size,size],[0,size]])
        M   = cv2.getPerspectiveTransform(self.corners, dst)
        return cv2.warpPerspective(self.gray, M, (size, size))


    def thick(self, min_width=5):
        mask = self.widths > min_width
        return self.ys[mask], self.centers[mask]


    def thin(self, max_width=3):
        mask = self.widths < max_width
        return self.ys[mask], self.centers[mask]


    def draw(self, warped=None):
        source = warped if warped is not None else self.gray
        output = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
        
        # # Continous lines through img
        # for y in np.unique(self.ys):
        #     cv2.line(output, (0, y), (source.shape[1], y), (0,255,0), 1)
        # for x in np.unique(self.centers):
        #     cv2.line(output, (x, 0), (x, source.shape[0]), (255,0,0), 1)
            
        # Vectorized — all at once
        if len(self.ys) > 0:
            output[self.ys, self.outers]  = [255,   0,   0]  # blue  = outer
            output[self.ys, self.inners]  = [  0,   0, 255]  # red   = inner
            output[self.ys, self.centers] = [  0, 255,   0]  # green = center

        # Draw corners
        if self.corners is not None:
            colors = [
                [255,   0,   0],   # top left     blue
                [  0, 255, 255],   # top right    yellow
                [  0,   0, 255],   # bottom right red
                [255,   0, 255]    # bottom left  purple
            ]
            
            for i, (cx, cy) in enumerate(self.corners):
                cv2.circle(output, (int(cx), int(cy)), 8, colors[i], -1)

        return output