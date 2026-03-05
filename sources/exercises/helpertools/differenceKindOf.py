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


#
# Traditional without numpy vectorized arrays #
#
# import cv2
# import numpy as np


# class Diff:

#     def __init__(self, img, diffThreshold=30, step=1, slots=1, slotThreshold=100):
#         self.img = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         self.diffThreshold = diffThreshold
#         self.step = step
#         self.slots = slots
#         self.slotThreshold = slotThreshold * 0.01
#         self.diff_arr = None
#         self.ys = []
#         self.outers = []
#         self.inners = []
#         self.centers = []
#         self.widths = []
#         self.corners = None

#     def _build_diff(self):
#         h, w = self.img.shape[:2]

#         self.diff_arr = [[0] * w for _ in range(h)]

#         for y in range(h):
#             for x in range(self.step, w - self.step):
#                 dx = int(self.img[y, x + self.step]) - int(self.img[y, x - self.step])
#                 dy = int(self.img[y + self.step, x]) - int(self.img[y - self.step, x]) if y >= self.step and y < h - self.step else 0
#                 self.diff_arr[y][x] = max(abs(dx), abs(dy))

#     # def _build_diff(self):
#     #     h, w = self.img.shape[:2]
#     #     self.diff_arr = [[0] * w for _ in range(h)]

#     #     for y in range(h):
#     #         for x in range(w):
#     #             dx = abs(int(self.img[y, x]) - int(self.img[y, x - self.step])) if x >= self.step else 0
#     #             dy = abs(int(self.img[y, x]) - int(self.img[y - self.step, x])) if y >= self.step else 0
#     #             self.diff_arr[y][x] = 1 if max(dx, dy) > self.diffThreshold else 0
    

#     #def _scan_line(self, row, y):
#         # w = len(row)
#         # in_edge = False

#         # for x in range(w):
#         #     if row[x] == 1 and not in_edge:
#         #         in_edge = True
#         #         outer = x
#         #     elif row[x] == 0 and in_edge:
#         #         in_edge = False
#         #         inner = x - 1
#         #         center = (outer + inner) // 2
#         #         width = inner - outer
#         #         self.ys.append(y)
#         #         self.outers.append(outer)
#         #         self.inners.append(inner)
#         #         self.centers.append(center)
#         #         self.widths.append(width)
                
#     def _scan_line(self, y):
#         w = self.gray.shape[1]
    
#         state = 'flat'
#         rising_start = 0
#         rising_sum = 0
#         edge_start = 0
    
#         for x in range(self.step, w - self.step):
#             diff = int(self.gray[y, x + self.step]) - int(self.gray[y, x - self.step])
        
#             if state == 'flat':
#                 if abs(diff) > self.diffThreshold:
#                     state = 'rising'
#                     rising_start = x
#                     rising_sum = abs(diff)

#             elif state == 'rising':
#                 if abs(diff) > self.diffThreshold:
#                     rising_sum += abs(diff)
#                 else:
#                     # Plateau — we're on the edge now
#                     state = 'edge'
#                     edge_start = x

#             elif state == 'edge':
#                 if abs(diff) > self.diffThreshold:
#                     # Falling edge detected
#                     state = 'falling'
#                     falling_sum = abs(diff)
#                     falling_start = x
            
#             elif state == 'falling':
#                 if abs(diff) > self.diffThreshold:
#                     falling_sum += abs(diff)
#                 else:
#                     # Edge complete — check symmetry and width
#                     width = falling_start - rising_start
#                     balance = abs(rising_sum - falling_sum) / max(rising_sum, falling_sum)
                
#                     if balance < 0.3 and width > 3:  # ~symmetric and wide enough
#                         outer  = rising_start
#                         inner  = falling_start
#                         center = (outer + inner) // 2
#                         self.ys.append(y)
#                         self.outers.append(outer)
#                         self.inners.append(inner)
#                         self.centers.append(center)
#                         self.widths.append(width)
                
#                     # Reset
#                     state = 'flat'
#                     rising_sum = 0


#     def detect(self):
        
#         self._build_diff()

#         self.ys = []
#         self.outers = []
#         self.inners = []
#         self.centers = []
#         self.widths = []

#         h = len(self.diff_arr)
#         for y in range(h):
#             self._scan_line(self.diff_arr[y], y)

#         self.corners = self._find_corners()
#         return self


#     def _find_corners(self):
#         if len(self.ys) < 4:
#             return None

#         h, w = self.img.shape
#         slot_h = h // self.slots
#         slot_w = w // self.slots

#         # Count edge points per slot
#         counts = [[0] * self.slots for _ in range(self.slots)]
#         for i in range(len(self.ys)):
#             sy = min(self.ys[i] // slot_h, self.slots - 1)
#             sx = min(self.centers[i] // slot_w, self.slots - 1)
#             counts[sy][sx] += 1

#         # Find max count across all slots
#         max_count = max(counts[r][c] for r in range(self.slots) for c in range(self.slots))
#         if max_count == 0:
#             return None

#         threshold = max_count * self.slotThreshold

#         # Keep only points in dense slots
#         dense_ys = []
#         dense_centers = []
#         for i in range(len(self.ys)):
#             sy = min(self.ys[i] // slot_h, self.slots - 1)
#             sx = min(self.centers[i] // slot_w, self.slots - 1)
#             if counts[sy][sx] >= threshold:
#                 dense_ys.append(self.ys[i])
#                 dense_centers.append(self.centers[i])

#         if len(dense_ys) < 4:
#             return None

#         # Find the 4 corners using sum/diff heuristic
#         top_left     = min(range(len(dense_ys)), key=lambda i: dense_ys[i] + dense_centers[i])
#         top_right    = min(range(len(dense_ys)), key=lambda i: dense_ys[i] - dense_centers[i])
#         bottom_right = max(range(len(dense_ys)), key=lambda i: dense_ys[i] + dense_centers[i])
#         bottom_left  = max(range(len(dense_ys)), key=lambda i: dense_ys[i] - dense_centers[i])

#         return np.float32([
#             [dense_centers[top_left],     dense_ys[top_left]],
#             [dense_centers[top_right],    dense_ys[top_right]],
#             [dense_centers[bottom_right], dense_ys[bottom_right]],
#             [dense_centers[bottom_left],  dense_ys[bottom_left]],
#         ])


#     def warp(self, size=500):
#         if self.corners is None:
#             return None
#         dst = np.float32([[0, 0], [size, 0], [size, size], [0, size]])
#         M = cv2.getPerspectiveTransform(self.corners, dst)
#         return cv2.warpPerspective(self.img, M, (size, size))


#     # def line_thickness(self, width=5):
#     #     ys = [self.ys[i] for i in range(len(self.ys)) if self.widths[i] > width]
#     #     cs = [self.centers[i] for i in range(len(self.ys)) if self.widths[i] > width]
#     #     return ys, cs


#     def draw(self, lineFill=1, cornerFill=5):
        
#         imgDraw = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR) if len(self.img == 2) else self.img
        
#         for i in range(len(self.ys)):
#             cv2.circle(imgDraw, (self.centers[i], self.ys[i]), lineFill, [0,0,255], -1)

#         if self.corners is not None:
#             colors = [
#                 [255, 0, 0],    # Top left blue
#                 [0, 255, 255],  # Top right yellow
#                 [0, 255, 0],    # Bottom right green
#                 [255, 255, 255],  # Bottom left white
#             ]
#             for i, (cx, cy) in enumerate(self.corners):
#                 cv2.circle(imgDraw, (int(cx), int(cy)), 8, colors[i], -1)

#         return imgDraw

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