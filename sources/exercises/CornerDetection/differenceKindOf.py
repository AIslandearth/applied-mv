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

    def __init__(self, gray, diffThreshold=30, step=1, slots=1, slotThreshold=100):
        self.gray = gray
        self.diffThreshold = diffThreshold
        self.step = step
        self.slots = slots
        self.slotThreshold = slotThreshold * 0.01 # From percentage to multiplier
        self.diff_arr = None
        self.ys = np.array([], dtype=int)
        self.outers = np.array([], dtype=int)
        self.inners = np.array([], dtype=int)
        self.centers = np.array([], dtype=int)
        self.widths = np.array([], dtype=int)
        self.corners = None


    def _build_diff(self):
        
        diff_x = np.abs(
            self.gray[:, self.step:].astype(np.int16) - self.gray[:, :-self.step].astype(np.int16)
        )
        diff_y = np.abs(
            self.gray[self.step:, :].astype(np.int16) - self.gray[:-self.step, :].astype(np.int16)
        )
        
        # np.pad(array, pad_width, mode)
        # pad_width order for 2D = ((top, bottom), (left, right))
        diff_x = np.pad(diff_x, ((0,0),(0,self.step)), mode='constant')
        diff_y = np.pad(diff_y, ((0,self.step),(0,0)), mode='constant')
        
        self.diff_arr = (np.maximum(diff_x, diff_y) > self.diffThreshold).astype(np.uint8)

    def _scan_line(self, line):
        
        diffs = np.diff(line.astype(np.int8))
        starts = np.where(diffs ==  1)[0]
        ends = np.where(diffs == -1)[0]
        n = min(len(starts), len(ends))
        if n == 0:
            return None
        outers = starts[:n]
        inners = ends[:n]
        return outers, inners, (outers+inners)//2, inners-outers


    # def _find_corners(self):
        
    #     # If ys is empty or ys and corners length is different or ys is smaller than 4 (min amount for perspective correction)
    #     if len(self.ys) == 0 or len(self.ys) < 4:
    #         return None

    #     points = np.column_stack([self.ys, self.centers])
    #     s = points[:, 0] + points[:, 1]
    #     d = points[:, 0] - points[:, 1]
    #     return np.float32([
    #         [points[s.argmin()][1], points[s.argmin()][0]],
    #         [points[d.argmin()][1], points[d.argmin()][0]],
    #         [points[s.argmax()][1], points[s.argmax()][0]],
    #         [points[d.argmax()][1], points[d.argmax()][0]]
    #     ])


    def detect(self):
        self._build_diff()

        h = self.diff_arr.shape[0]
        all_ys = []
        all_outers = []
        all_inners = []

        # Scan all rows
        y = 0
        while y < h:
            result = self._scan_line(self.diff_arr[y, :])
            if result is not None:
                outers, inners, _, _ = result
                all_ys.extend([y]  * len(outers))
                all_outers.extend(outers)
                all_inners.extend(inners)
            y += 1

        self.ys      = np.array(all_ys)
        self.outers  = np.array(all_outers)
        self.inners  = np.array(all_inners)
        # // Trunc divide
        self.centers = (self.outers + self.inners) // 2
        self.widths  = self.inners - self.outers
        self.corners = self._find_corners()

        return self
    
    def _find_corners(self):
        
        if len(self.ys) == 0 or len(self.ys) < 4:
            return None

        h, w = self.gray.shape

        # Determine slot/grid amount
        slot_h = h // self.slots
        slot_w = w // self.slots

        # Count edge points per slot
        counts = np.zeros((self.slots, self.slots), dtype=int)

        sy = self.ys      // slot_h   # which slot row each point belongs to
        sx = self.centers // slot_w   # which slot col each point belongs to

        # Clip to valid range — points at edge may go out of bounds
        sy = np.clip(sy, 0, (self.slots - 1))
        sx = np.clip(sx, 0, (self.slots - 1))

        # Count per slot — vectorized
        for idx in range(len(self.ys)):
            counts[sy[idx]][sx[idx]] += 1

        # Threshold for slot/grid density
        max_count = counts.max()
        if max_count == 0:
            return None

        # keep slots with >20% of max
        thresholdTemp = max_count * self.slotThreshold

        # Keep only the dense "slots"
        dense_mask        = counts[sy, sx] >= thresholdTemp

        dense_ys          = self.ys[dense_mask]
        dense_centers     = self.centers[dense_mask]

        if len(dense_ys) < 4:
            return None

        # Dense points only
        points = np.column_stack([dense_ys, dense_centers])
        s      = points[:, 0] + points[:, 1]
        d      = points[:, 0] - points[:, 1]

        return np.float32([
            [points[s.argmin()][1], points[s.argmin()][0]],
            [points[d.argmin()][1], points[d.argmin()][0]],
            [points[s.argmax()][1], points[s.argmax()][0]],
            [points[d.argmax()][1], points[d.argmax()][0]]
        ])


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
        
        # Vectorized — all at once
        if len(self.ys) > 0:
            output[self.ys, self.outers] = [255, 0, 0]  # blue = outer
            output[self.ys, self.inners] = [0, 255, 0]  # green = inner
            output[self.ys, self.centers] = [0, 0, 255]  # red = center
            
        # Fill between outer and inner per detected edge
        for oi in range(len(self.ys)):
            y     = self.ys[oi]
            outer = self.outers[oi]
            inner = self.inners[oi]
            output[y, outer:inner] = [0, 0, 255]   # red = fil btwn outer and inner

        # Draw corners
        if self.corners is not None:
            colors = [
                [255, 0, 0],   # top left     blue
                [0, 255, 255],   # top right    yellow
                [0, 0, 255],   # bottom right red
                [255, 0, 255]    # bottom left  purple
            ]
            
            for i, (cx, cy) in enumerate(self.corners):
                cv2.circle(output, (int(cx), int(cy)), 8, colors[i], -1)

        return output