import cv2
import numpy as np

# class Diff:

#     def __init__(self, gray, diffThreshold=30, step=1, slots=1, slotThreshold=100):
#         self.gray = gray
#         self.diffThreshold = diffThreshold
#         self.step = step
#         self.slots = slots
#         self.slotThreshold = slotThreshold * 0.01 # From percentage to multiplier
#         self.diff_arr = None
#         self.ys = np.array([], dtype=int)
#         self.outers = np.array([], dtype=int)
#         self.inners = np.array([], dtype=int)
#         self.centers = np.array([], dtype=int)
#         self.widths = np.array([], dtype=int)
#         self.corners = None


#     def _build_diff(self):
        
#         diff_x = np.abs(
#             self.gray[:, self.step:].astype(np.int16) - self.gray[:, :-self.step].astype(np.int16)
#         )
#         diff_y = np.abs(
#             self.gray[self.step:, :].astype(np.int16) - self.gray[:-self.step, :].astype(np.int16)
#         )
        
#         # np.pad(array, pad_width, mode)
#         # pad_width order for 2D = ((top, bottom), (left, right))
#         diff_x = np.pad(diff_x, ((0,0),(0,self.step)), mode='constant')
#         diff_y = np.pad(diff_y, ((0,self.step),(0,0)), mode='constant')
        
#         self.diff_arr = (np.maximum(diff_x, diff_y) > self.diffThreshold).astype(np.uint8)

#     def _scan_line(self, line):
        
#         diffs = np.diff(line.astype(np.int8))
#         starts = np.where(diffs ==  1)[0]
#         ends = np.where(diffs == -1)[0]
#         n = min(len(starts), len(ends))
#         if n == 0:
#             return None
#         outers = starts[:n]
#         inners = ends[:n]
#         return outers, inners, (outers+inners)//2, inners-outers


#     # def _find_corners(self):
        
#     #     # If ys is empty or ys and corners length is different or ys is smaller than 4 (min amount for perspective correction)
#     #     if len(self.ys) == 0 or len(self.ys) < 4:
#     #         return None

#     #     points = np.column_stack([self.ys, self.centers])
#     #     s = points[:, 0] + points[:, 1]
#     #     d = points[:, 0] - points[:, 1]
#     #     return np.float32([
#     #         [points[s.argmin()][1], points[s.argmin()][0]],
#     #         [points[d.argmin()][1], points[d.argmin()][0]],
#     #         [points[s.argmax()][1], points[s.argmax()][0]],
#     #         [points[d.argmax()][1], points[d.argmax()][0]]
#     #     ])


#     def detect(self):
#         self._build_diff()

#         h = self.diff_arr.shape[0]
#         all_ys = []
#         all_outers = []
#         all_inners = []

#         # Scan all rows
#         y = 0
#         while y < h:
#             result = self._scan_line(self.diff_arr[y, :])
#             if result is not None:
#                 outers, inners, _, _ = result
#                 all_ys.extend([y]  * len(outers))
#                 all_outers.extend(outers)
#                 all_inners.extend(inners)
#             y += 1

#         self.ys      = np.array(all_ys)
#         self.outers  = np.array(all_outers)
#         self.inners  = np.array(all_inners)
#         # // Trunc divide
#         self.centers = (self.outers + self.inners) // 2
#         self.widths  = self.inners - self.outers
#         self.corners = self._find_corners()

#         return self
    
#     def _find_corners(self):
        
#         if len(self.ys) == 0 or len(self.ys) < 4:
#             return None

#         h, w = self.gray.shape

#         # Determine slot/grid amount
#         slot_h = h // self.slots
#         slot_w = w // self.slots

#         # Count edge points per slot
#         counts = np.zeros((self.slots, self.slots), dtype=int)

#         sy = self.ys      // slot_h   # which slot row each point belongs to
#         sx = self.centers // slot_w   # which slot col each point belongs to

#         # Clip to valid range — points at edge may go out of bounds
#         sy = np.clip(sy, 0, (self.slots - 1))
#         sx = np.clip(sx, 0, (self.slots - 1))

#         # Count per slot — vectorized
#         for idx in range(len(self.ys)):
#             counts[sy[idx]][sx[idx]] += 1

#         # Threshold for slot/grid density
#         max_count = counts.max()
#         if max_count == 0:
#             return None

#         # keep slots with >20% of max
#         thresholdTemp = max_count * self.slotThreshold

#         # Keep only the dense "slots"
#         dense_mask        = counts[sy, sx] >= thresholdTemp

#         dense_ys          = self.ys[dense_mask]
#         dense_centers     = self.centers[dense_mask]

#         if len(dense_ys) < 4:
#             return None

#         # Dense points only
#         points = np.column_stack([dense_ys, dense_centers])
#         s      = points[:, 0] + points[:, 1]
#         d      = points[:, 0] - points[:, 1]

#         return np.float32([
#             [points[s.argmin()][1], points[s.argmin()][0]],
#             [points[d.argmin()][1], points[d.argmin()][0]],
#             [points[s.argmax()][1], points[s.argmax()][0]],
#             [points[d.argmax()][1], points[d.argmax()][0]]
#         ])


#     def warp(self, size=500):
#         if self.corners is None:
#             return None
#         dst = np.float32([[0,0],[size,0],[size,size],[0,size]])
#         M   = cv2.getPerspectiveTransform(self.corners, dst)
#         return cv2.warpPerspective(self.gray, M, (size, size))


#     def thick(self, min_width=5):
#         mask = self.widths > min_width
#         return self.ys[mask], self.centers[mask]


#     def thin(self, max_width=3):
#         mask = self.widths < max_width
#         return self.ys[mask], self.centers[mask]


#     def draw(self, warped=None):
#         source = warped if warped is not None else self.gray
#         output = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
        
#         # Vectorized — all at once
#         if len(self.ys) > 0:
#             output[self.ys, self.outers] = [255, 0, 0]  # blue = outer
#             output[self.ys, self.inners] = [0, 255, 0]  # green = inner
#             output[self.ys, self.centers] = [0, 0, 255]  # red = center
            
#         # Fill between outer and inner per detected edge
#         for oi in range(len(self.ys)):
#             y     = self.ys[oi]
#             outer = self.outers[oi]
#             inner = self.inners[oi]
#             output[y, outer:inner] = [0, 0, 255]   # red = fil btwn outer and inner

#         # Draw corners
#         if self.corners is not None:
#             colors = [
#                 [255, 0, 0],   # top left     blue
#                 [0, 255, 255],   # top right    yellow
#                 [0, 0, 255],   # bottom right red
#                 [255, 0, 255]    # bottom left  purple
#             ]
            
#             for i, (cx, cy) in enumerate(self.corners):
#                 cv2.circle(output, (int(cx), int(cy)), 8, colors[i], -1)

#         return output
    
    
    
    
    
    
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
    
    
    
    
    
    
# import cv2
# import numpy as np


# class Diff:

#     def __init__(self, img, diffThreshold=30, step=1, slots=1, slotThreshold=100):
#         self.img          = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         self.diffThreshold = diffThreshold
#         self.step          = step
#         self.slots         = slots
#         self.slotThreshold = slotThreshold * 0.01
#         self.ys            = []
#         self.outers        = []
#         self.inners        = []
#         self.centers       = []
#         self.widths        = []
#         self.corners       = None


#     def _scan_line(self, y):
#         w             = self.img.shape[1]
#         state         = 'no line'
#         rising_edge  = 0
#         rising_sum    = 0
#         falling_sum   = 0
#         falling_edge = 0

#         for x in range(self.step, w - self.step):
#             diff = abs(int(self.img[y, x + self.step]) - int(self.img[y, x - self.step]))

#             if state == 'no line':
#                 if diff > self.diffThreshold:
#                     state        = 'rising'
#                     rising_edge = x
#                     rising_sum   = diff

#             elif state == 'rising':
#                 if diff > self.diffThreshold:
#                     rising_sum += diff
#                 else:
#                     state = 'edge'

#             elif state == 'edge':
#                 if diff > self.diffThreshold:
#                     state         = 'falling'
#                     falling_edge = x
#                     falling_sum   = diff

#             elif state == 'falling':
#                 if diff > self.diffThreshold:
#                     falling_sum += diff
#                 else:
#                     width   = falling_edge - rising_edge
#                     balance = abs(rising_sum - falling_sum) / max(rising_sum, falling_sum)

#                     if balance < 0.1 and width > 20:
#                         self.ys.append(y)
#                         self.outers.append(rising_edge)
#                         self.inners.append(falling_edge)
#                         self.centers.append((rising_edge + falling_edge) // 2)
#                         self.widths.append(width)

#                     state        = 'flat'
#                     rising_sum   = 0
#                     falling_sum  = 0


#     def detect(self):
#         h, w = self.img.shape[:2]

#         for y in range(h):
#             self._scan_line(y)
#             for x in range(w):
#                 self._scan_line(x)

#         self.corners = self._find_corners()
#         return self


#     def _find_corners(self):
#         if len(self.ys) < 4:
#             return None

#         h, w   = self.img.shape
#         slot_h = h // self.slots
#         slot_w = w // self.slots

#         counts = [[0] * self.slots for _ in range(self.slots)]
#         for i in range(len(self.ys)):
#             sy = min(self.ys[i]      // slot_h, self.slots - 1)
#             sx = min(self.centers[i] // slot_w, self.slots - 1)
#             counts[sy][sx] += 1

#         max_count = max(counts[r][c] for r in range(self.slots) for c in range(self.slots))
#         if max_count == 0:
#             return None

#         threshold   = max_count * self.slotThreshold
#         dense_ys    = []
#         dense_centers = []

#         for i in range(len(self.ys)):
#             sy = min(self.ys[i]      // slot_h, self.slots - 1)
#             sx = min(self.centers[i] // slot_w, self.slots - 1)
#             if counts[sy][sx] >= threshold:
#                 dense_ys.append(self.ys[i])
#                 dense_centers.append(self.centers[i])

#         if len(dense_ys) < 4:
#             return None

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
#         M   = cv2.getPerspectiveTransform(self.corners, dst)
#         return cv2.warpPerspective(self.img, M, (size, size))


#     def draw(self, lineFill=1, cornerFill=8):
#         output = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR) if len(self.img.shape) == 2 else self.img.copy()

#         for i in range(len(self.ys)):
#             cv2.circle(output, (self.centers[i], self.ys[i]), lineFill, [0, 0, 255], -1)

#         if self.corners is not None:
#             colors = [
#                 [255, 0,   0  ],  # top left     blue
#                 [0,   255, 255],  # top right    yellow
#                 [0,   255, 0  ],  # bottom right green
#                 [255, 255, 255],  # bottom left  white
#             ]
#             for i, (cx, cy) in enumerate(self.corners):
#                 cv2.circle(output, (int(cx), int(cy)), cornerFill, colors[i], -1)

#         return output
    
    
# class Diff:

#     def __init__(self, img,
#                  diff_threshold=30,
#                  step=1,
#                  min_width=20,
#                  max_width=None,
#                  slot_threshold=0.5):

#         self.img            = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         self.diff_threshold = diff_threshold
#         self.step           = step
#         self.min_width      = min_width
#         self.max_width      = max_width or img.shape[1]
#         self.slot_threshold = slot_threshold

#         # Populated after detect()
#         self.h_hits  = []   # (x_center, y)  from horizontal scan
#         self.v_hits  = []   # (x, y_center)  from vertical scan
#         self.hits    = []   # combined (x, y) after intersection
#         self.corners = None


#     # ------------------------------------------------------------------
#     # Gradient helpers
#     # ------------------------------------------------------------------

#     def _gradient(self, line, pos):
#         """Symmetric gradient at pos using self.step."""
#         return abs(int(line[pos + self.step]) - int(line[pos - self.step]))


#     # ------------------------------------------------------------------
#     # Single-line scan  (one 1-D array, left → right)
#     # Returns list of (center, sum) for every edge pair found
#     # ------------------------------------------------------------------

#     def _scan_line(self, line):
#         results = []
#         n       = len(line)
#         x       = self.step

#         while x < n - self.step:
#             # Look for a rising edge
#             if self._gradient(line, x) <= self.diff_threshold:
#                 x += 1
#                 continue

#             # Accumulate rising-edge region
#             rise_start = x
#             rise_sum   = 0
#             while x < n - self.step and self._gradient(line, x) > self.diff_threshold:
#                 rise_sum += self._gradient(line, x)
#                 x        += 1

#             # Flat region between edges (skip it)
#             flat_start = x
#             while x < n - self.step and self._gradient(line, x) <= self.diff_threshold:
#                 x += 1
#             if x == flat_start:             # no flat gap → not a clean edge pair
#                 continue

#             # Accumulate falling-edge region
#             fall_start = x
#             fall_sum   = 0
#             while x < n - self.step and self._gradient(line, x) > self.diff_threshold:
#                 fall_sum += self._gradient(line, x)
#                 x        += 1

#             if fall_sum == 0:               # never found a falling edge
#                 continue

#             width   = fall_start - rise_start
#             balance = abs(rise_sum - fall_sum) / max(rise_sum, fall_sum)

#             if (balance < 0.2
#                     and self.min_width <= width <= self.max_width):
#                 center = (rise_start + fall_start) // 2
#                 results.append((center, rise_sum + fall_sum))

#         return results


#     # ------------------------------------------------------------------
#     # Full detect: horizontal rows then vertical columns, then combine
#     # ------------------------------------------------------------------

#     def detect(self):
#         h, w = self.img.shape[:2]

#         # Horizontal pass: scan every row → hits are (x_center, y)
#         for y in range(h):
#             for cx, _ in self._scan_line(self.img[y, :]):
#                 self.h_hits.append((cx, y))

#         # Vertical pass: scan every column → hits are (x, y_center)
#         for x in range(w):
#             for cy, _ in self._scan_line(self.img[:, x]):
#                 self.v_hits.append((x, cy))

#         # Intersect: keep only points agreed on by both passes
#         # (within ±step tolerance so we don't miss by one pixel)
#         h_set = {(cx, y) for cx, y in self.h_hits}
#         tol   = self.step + 1

#         for vx, vy in self.v_hits:
#             for dx in range(-tol, tol + 1):
#                 for dy in range(-tol, tol + 1):
#                     if (vx + dx, vy + dy) in h_set:
#                         self.hits.append((vx, vy))
#                         break
#                 else:
#                     continue
#                 break

#         self.corners = self._find_corners()
#         return self


#     # ------------------------------------------------------------------
#     # Corner detection from combined hits
#     # ------------------------------------------------------------------

#     def _find_corners(self):
#         if len(self.hits) < 4:
#             return None

#         h, w = self.img.shape[:2]

#         # Density filter: divide image into a grid, keep hits in
#         # slots that reach slot_threshold × max_slot_count
#         slots  = 4
#         slot_h = max(h // slots, 1)
#         slot_w = max(w // slots, 1)

#         counts = {}
#         for x, y in self.hits:
#             key = (min(y // slot_h, slots - 1), min(x // slot_w, slots - 1))
#             counts[key] = counts.get(key, 0) + 1

#         if not counts:
#             return None

#         max_count = max(counts.values())
#         threshold = max_count * self.slot_threshold

#         dense = [(x, y) for x, y in self.hits
#                  if counts[(min(y // slot_h, slots - 1),
#                             min(x // slot_w, slots - 1))] >= threshold]

#         if len(dense) < 4:
#             return None

#         # Four extreme points by diagonal score
#         top_left     = min(dense, key=lambda p: p[0] + p[1])
#         top_right    = min(dense, key=lambda p: p[1] - p[0])
#         bottom_right = max(dense, key=lambda p: p[0] + p[1])
#         bottom_left  = max(dense, key=lambda p: p[1] - p[0])

#         return np.float32([top_left, top_right, bottom_right, bottom_left])


#     # ------------------------------------------------------------------
#     # Perspective warp
#     # ------------------------------------------------------------------

#     def warp(self, size=500):
#         if self.corners is None:
#             return None
#         dst = np.float32([[0, 0], [size, 0], [size, size], [0, size]])
#         M   = cv2.getPerspectiveTransform(self.corners, dst)
#         return cv2.warpPerspective(self.img, M, (size, size))


#     # ------------------------------------------------------------------
#     # Debug visualisation
#     # ------------------------------------------------------------------

#     def draw(self, line_r=1, corner_r=8):
#         out = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)

#         for x, y in self.h_hits:
#             cv2.circle(out, (x, y), line_r, (0, 80, 255), -1)   # faint orange

#         for x, y in self.v_hits:
#             cv2.circle(out, (x, y), line_r, (255, 80, 0), -1)   # faint blue

#         for x, y in self.hits:
#             cv2.circle(out, (x, y), line_r + 1, (0, 255, 0), -1) # green = confirmed

#         if self.corners is not None:
#             colors = [(255, 0, 0), (0, 255, 255), (0, 255, 0), (255, 255, 255)]
#             for (cx, cy), color in zip(self.corners, colors):
#                 cv2.circle(out, (int(cx), int(cy)), corner_r, color, -1)

#         return out






# class Diff:

#     def __init__(self, gray, threshold=30, step=2,
#                  min_width=2,  max_width=50,
#                  min_gray=0,   max_gray=200,
#                  fill=False, slots=1, density=100):

#         self.gray      = gray
#         self.threshold = threshold   # difference sensitivity — adaptive
#         self.step      = step        # blur + lookup distance — one param
#         self.min_width = min_width   # min line width px
#         self.max_width = max_width   # max line width px
#         self.min_gray  = min_gray    # min gray value at center
#         self.max_gray  = max_gray    # max gray value at center
#         self.fill      = fill        # True=fill, False=center only
#         self.slots = slots # Slicing to grids
#         self.density = density * 0.01 #

#         self.diff_arr  = None        # combined y and x difference array, height and width
#         self.ys        = np.array([], dtype=int)
#         self.outers    = np.array([], dtype=int)
#         self.inners    = np.array([], dtype=int)
#         self.centers   = np.array([], dtype=int)
#         self.widths    = np.array([], dtype=int)
#         self.corners   = None


#     def _build_diff(self):
#         # Horizontal difference, step as blur + lookup
#         diff_x = np.abs(
#             self.gray[:, self.step:].astype(np.int16) - self.gray[:, :-self.step].astype(np.int16)
#         )
#         # Vertical difference, step as blur + lookup
#         diff_y = np.abs(
#             self.gray[self.step:, :].astype(np.int16) - self.gray[:-self.step, :].astype(np.int16)
#         )

#         # Pad back to original size
#         diff_x = np.pad(diff_x, ((0, 0),         (0, self.step)), mode='constant')
#         diff_y = np.pad(diff_y, ((0, self.step), (0, 0)),         mode='constant')

#         # Combine x and y direction arrays, significant change either direction = edge
#         # Threshold based on difference
#         self.diff_arr = (np.maximum(diff_x, diff_y) > self.threshold).astype(np.uint8)


#     def _scan_line(self, line):
#         # Rising edge -> on line -> falling edge
#         diffs  = np.diff(line.astype(np.int8))
#         starts = np.where(diffs ==  1)[0]   # rising  edge
#         ends   = np.where(diffs == -1)[0]   # falling edge
#         n      = min(len(starts), len(ends))
#         if n == 0:
#             return None
#         outers = starts[:n]
#         inners = ends[:n]
#         return outers, inners, (outers + inners) // 2, inners - outers


#     def _filter(self):
#         if len(self.ys) == 0:
#             return

#         # Width filter, within min/max range
#         width_mask = (self.widths >= self.min_width) & \
#                      (self.widths <= self.max_width)

#         # "Color" value filter
#         # Check original grayscale at detected center
#         # Adaptive, relative to actual image content
#         gray_vals  = self.gray[self.ys, self.centers]
#         gray_mask  = (gray_vals >= self.min_gray) & \
#                      (gray_vals <= self.max_gray)

#         # Density filter, n slots
#         h, w   = self.gray.shape
#         slot_h = max(1, h // self.slots)
#         slot_w = max(1, w // self.slots)

#         sy = np.clip(self.ys      // slot_h, 0, self.slots)
#         sx = np.clip(self.centers // slot_w, 0, self.slots)

#         # Count points per slot — vectorized
#         counts    = np.zeros((self.slots, self.slots), dtype=int)
#         np.add.at(counts, (sy, sx), 1)   # vectorized count

#         max_count  = counts.max()
#         if max_count > 0:
#             density_mask = counts[sy, sx] >= max_count * self.slots
#         else:
#             density_mask = np.ones(len(self.ys), dtype=bool)

#         # All filters combined
#         mask         = width_mask & gray_mask & density_mask
#         self.ys      = self.ys[mask]
#         self.outers  = self.outers[mask]
#         self.inners  = self.inners[mask]
#         self.centers = self.centers[mask]
#         self.widths  = self.widths[mask]


#     def _find_corners(self):
#         if len(self.ys) < 4:
#             return None

#         points = np.column_stack([self.ys, self.centers])
#         s      = points[:, 0] + points[:, 1]
#         d      = points[:, 0] - points[:, 1]

#         return np.float32([
#             [points[s.argmin()][1], points[s.argmin()][0]],
#             [points[d.argmin()][1], points[d.argmin()][0]],
#             [points[s.argmax()][1], points[s.argmax()][0]],
#             [points[d.argmax()][1], points[d.argmax()][0]]
#         ])


#     def detect(self):
#         # Adaptive difference array y+x combined
#         self._build_diff()

#         h          = self.diff_arr.shape[0]
#         all_ys     = []
#         all_outers = []
#         all_inners = []

#         # Scan every row, rising/falling edges
#         y = 0
#         while y < h:
#             result = self._scan_line(self.diff_arr[y, :])
#             if result is not None:
#                 outers, inners, centers, widths = result
#                 all_ys.extend([y] * len(outers))
#                 all_outers.extend(outers)
#                 all_inners.extend(inners)
#             y += 1

#         self.ys      = np.array(all_ys)
#         self.outers  = np.array(all_outers)
#         self.inners  = np.array(all_inners)
#         self.centers = (self.outers + self.inners) // 2
#         self.widths  = self.inners - self.outers

#         # Filter: width + gray + density all together
#         self._filter()

#         # Corners from filtered points only
#         self.corners = self._find_corners()

#         return self


#     def warp(self, size=500):
#         if self.corners is None:
#             return None
#         dst = np.float32([[0,0],[size,0],[size,size],[0,size]])
#         M   = cv2.getPerspectiveTransform(self.corners, dst)
#         return cv2.warpPerspective(self.gray, M, (size, size))


#     def draw(self, warped=None):
#         source = warped if warped is not None else self.gray
#         output = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)

#         if len(self.ys) > 0:
#             if self.fill:
#                 # Fill outer -> inner -> full line width
#                 for idx in range(len(self.ys)):
#                     #output[self.ys[idx], self.outers[idx]:self.inners[idx]] = [0, 255, 255]
#                     output[self.ys, self.centers] = [0, 0, 255]   # red   = center on top
#             else:
#                 output[self.ys, self.centers] = [0, 0, 255]   # red   = center only

#             #output[self.ys, self.outers] = [255, 0,   0]      # blue  = outer edge
#             #output[self.ys, self.inners] = [0,   255, 0]      # green = inner edge

#         if self.corners is not None:
#             colors = [[255,0,0],[0,255,255],[0,0,255],[255,0,255]]
#             for i, (cx, cy) in enumerate(self.corners):
#                 cv2.circle(output, (int(cx), int(cy)), 8, colors[i], -1)

#         return output
    
    
    
    
    


# import cv2
# import numpy as np


# class Diff:

#     def __init__(self, img, diff_threshold=30, step=2,
#                  min_width=2,   max_width=50,
#                  min_gray=0,    max_gray=200,
#                  balance=0.3,   slots=9,
#                  slot_threshold=50, fill=False):

#         # Accept color or gray
#         self.gray           = img if len(img.shape) == 2 \
#                               else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         self.diff_threshold = diff_threshold   # adaptive — relative change
#         self.step           = step             # blur + lookup distance
#         self.min_width      = min_width        # min line width px
#         self.max_width      = max_width        # max line width px
#         self.min_gray       = min_gray         # min gray at center
#         self.max_gray       = max_gray         # max gray at center
#         self.balance        = balance          # rising/falling symmetry 0-1
#         self.slots          = slots            # density grid size
#         self.slot_threshold = slot_threshold   # density % 0-100
#         self.fill           = fill             # True=fill, False=center only

#         self.ys      = []
#         self.outers  = []
#         self.inners  = []
#         self.centers = []
#         self.widths  = []
#         self.corners = None


#     def _scan_line(self, line, line_idx, is_row=True):
#         n             = len(line)
#         state         = 'flat'
#         rising_edge   = 0
#         falling_edge  = 0
#         rising_sum    = 0
#         falling_sum   = 0

#         x = self.step
#         while x < n - self.step:

#             # Difference — step as blur + lookup
#             # Adaptive — relative change not absolute value
#             diff = abs(int(line[x + self.step]) - int(line[x - self.step]))

#             if state == 'flat':
#                 if diff > self.diff_threshold:
#                     state       = 'rising'
#                     rising_edge = x
#                     rising_sum  = diff

#             elif state == 'rising':
#                 if diff > self.diff_threshold:
#                     rising_sum += diff       # accumulate rising strength
#                 else:
#                     state = 'on_line'        # entered the line

#             elif state == 'on_line':
#                 if diff > self.diff_threshold:
#                     state        = 'falling'
#                     falling_edge = x
#                     falling_sum  = diff

#             elif state == 'falling':
#                 if diff > self.diff_threshold:
#                     falling_sum += diff      # accumulate falling strength
#                 else:
#                     # ── Complete edge pair found ─────────
#                     width   = falling_edge - rising_edge
#                     center  = (rising_edge + falling_edge) // 2

#                     # Balance — rising and falling should match
#                     # symmetric edge = real line
#                     # asymmetric     = noise
#                     sym = abs(rising_sum - falling_sum) / \
#                           max(rising_sum, falling_sum)

#                     # Gray value at center — original image
#                     if is_row:
#                         gray_val = int(self.gray[line_idx, center])
#                     else:
#                         gray_val = int(self.gray[center, line_idx])

#                     # All filters — width, balance, gray
#                     if (self.min_width <= width  <= self.max_width and
#                         sym            <= self.balance             and
#                         self.min_gray  <= gray_val <= self.max_gray):

#                         if is_row:
#                             self.ys.append(line_idx)
#                             self.outers.append(rising_edge)
#                             self.inners.append(falling_edge)
#                             self.centers.append(center)
#                             self.widths.append(width)
#                         else:
#                             self.ys.append(rising_edge)
#                             self.outers.append(line_idx)
#                             self.inners.append(line_idx)
#                             self.centers.append(center)
#                             self.widths.append(width)

#                     # Reset — look for next line
#                     state       = 'flat'
#                     rising_sum  = 0
#                     falling_sum = 0

#             x += 1


#     def _find_corners(self):
#         if len(self.ys) < 4:
#             return None

#         h, w   = self.gray.shape
#         slot_h = max(1, h // self.slots)
#         slot_w = max(1, w // self.slots)

#         # Count per slot
#         counts = [[0] * self.slots for _ in range(self.slots)]
#         for i in range(len(self.ys)):
#             sy = min(self.ys[i]      // slot_h, self.slots - 1)
#             sx = min(self.centers[i] // slot_w, self.slots - 1)
#             counts[sy][sx] += 1

#         max_count = max(counts[r][c]
#                         for r in range(self.slots)
#                         for c in range(self.slots))
#         if max_count == 0:
#             return None

#         threshold = max_count * (self.slot_threshold * 0.01)

#         # Keep only points in dense slots
#         dense_ys      = []
#         dense_centers = []
#         for i in range(len(self.ys)):
#             sy = min(self.ys[i]      // slot_h, self.slots - 1)
#             sx = min(self.centers[i] // slot_w, self.slots - 1)
#             if counts[sy][sx] >= threshold:
#                 dense_ys.append(self.ys[i])
#                 dense_centers.append(self.centers[i])

#         if len(dense_ys) < 4:
#             return None

#         # Corners — sum/diff math
#         top_left     = min(range(len(dense_ys)),
#                            key=lambda i: dense_ys[i] + dense_centers[i])
#         top_right    = min(range(len(dense_ys)),
#                            key=lambda i: dense_ys[i] - dense_centers[i])
#         bottom_right = max(range(len(dense_ys)),
#                            key=lambda i: dense_ys[i] + dense_centers[i])
#         bottom_left  = max(range(len(dense_ys)),
#                            key=lambda i: dense_ys[i] - dense_centers[i])

#         return np.float32([
#             [dense_centers[top_left],     dense_ys[top_left]    ],
#             [dense_centers[top_right],    dense_ys[top_right]   ],
#             [dense_centers[bottom_right], dense_ys[bottom_right]],
#             [dense_centers[bottom_left],  dense_ys[bottom_left] ],
#         ])


#     def detect(self):
#         h, w = self.gray.shape

#         # Scan all rows — horizontal
#         y = 0
#         while y < h:
#             self._scan_line(self.gray[y, :], y, is_row=True)
#             y += 1

#         # Scan all columns — vertical
#         x = 0
#         while x < w:
#             self._scan_line(self.gray[:, x], x, is_row=False)
#             x += 1

#         self.corners = self._find_corners()
#         return self


#     def warp(self, size=500):
#         if self.corners is None:
#             return None
#         dst = np.float32([[0,0],[size,0],[size,size],[0,size]])
#         M   = cv2.getPerspectiveTransform(self.corners, dst)
#         return cv2.warpPerspective(self.gray, M, (size, size))


#     def draw(self, warped=None):
#         source = warped if warped is not None else self.gray
#         output = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR) \
#                  if len(source.shape) == 2 else source.copy()

#         for i in range(len(self.ys)):
#             if self.fill:
#                 # Fill outer → inner
#                 cv2.line(output,
#                          (self.outers[i],  self.ys[i]),
#                          (self.inners[i],  self.ys[i]),
#                          [0, 255, 255], 1)

#             # Center point
#             cv2.circle(output,
#                        (self.centers[i], self.ys[i]),
#                        1, [0, 0, 255], -1)

#             # Outer + inner edges
#             output[self.ys[i], self.outers[i]] = [255, 0,   0]
#             output[self.ys[i], self.inners[i]] = [0,   255, 0]

#         if self.corners is not None:
#             colors = [[255,0,0],[0,255,255],[0,255,0],[255,255,255]]
#             for i, (cx, cy) in enumerate(self.corners):
#                 cv2.circle(output, (int(cx), int(cy)), 8, colors[i], -1)

#         return output
    
    

class Diff:

    def __init__(self, gray, threshold=30, step=2, min_width=2, max_width=50):
        self.gray      = gray
        self.threshold = threshold
        self.step      = step
        self.min_width = min_width
        self.max_width = max_width
        self.ys        = np.array([], dtype=int)
        self.outers    = np.array([], dtype=int)
        self.inners    = np.array([], dtype=int)
        self.centers   = np.array([], dtype=int)
        self.widths    = np.array([], dtype=int)
        self.corners   = None


    def _build_diff(self):
        # One pass — difference array H+V combined
        diff_x = np.abs(
            self.gray[:, self.step:].astype(np.int16) -
            self.gray[:, :-self.step].astype(np.int16)
        )
        diff_y = np.abs(
            self.gray[self.step:, :].astype(np.int16) -
            self.gray[:-self.step, :].astype(np.int16)
        )
        diff_x = np.pad(diff_x, ((0,0),(0,self.step)),         mode='constant')
        diff_y = np.pad(diff_y, ((0,self.step),(0,0)),         mode='constant')
        return (np.maximum(diff_x, diff_y) > self.threshold).astype(np.uint8)


    def _scan_line(self, line):
        # Rising edge → inner → falling edge → center
        diffs  = np.diff(line.astype(np.int8))
        starts = np.where(diffs ==  1)[0]   # rising  edges
        ends   = np.where(diffs == -1)[0]   # falling edges
        n      = min(len(starts), len(ends))
        if n == 0:
            return None
        outers  = starts[:n]
        inners  = ends[:n]
        centers = (outers + inners) // 2
        widths  = inners - outers
        # Width filter
        mask    = (widths >= self.min_width) & (widths <= self.max_width)
        if not np.any(mask):
            return None
        return outers[mask], inners[mask], centers[mask], widths[mask]


    def _find_corners(self):
        if len(self.ys) < 4:
            return None
        h, w   = self.gray.shape
        slot_h = max(1, h // 9)
        slot_w = max(1, w // 9)
        sy     = np.clip(self.ys      // slot_h, 0, 8)
        sx     = np.clip(self.centers // slot_w, 0, 8)
        counts = np.zeros((9, 9), dtype=int)
        np.add.at(counts, (sy, sx), 1)
        mask          = counts[sy, sx] >= counts.max() * 0.5
        ys, centers   = self.ys[mask], self.centers[mask]
        if len(ys) < 4:
            return None
        pts = np.column_stack([ys, centers])
        s   = pts[:, 0] + pts[:, 1]
        d   = pts[:, 0] - pts[:, 1]
        return np.float32([
            [pts[s.argmin()][1], pts[s.argmin()][0]],
            [pts[d.argmin()][1], pts[d.argmin()][0]],
            [pts[s.argmax()][1], pts[s.argmax()][0]],
            [pts[d.argmax()][1], pts[d.argmax()][0]]
        ])


    def detect(self):
        diff       = self._build_diff()
        h          = diff.shape[0]
        all_ys, all_outers, all_inners = [], [], []

        y = 0
        while y < h:
            result = self._scan_line(diff[y, :])
            if result is not None:
                o, i, c, w = result
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


    def draw(self, warped=None):
        source = warped if warped is not None else self.gray
        output = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
        if len(self.ys) > 0:
            output[self.ys, self.outers]  = [255,   0,   0]  # blue  = outer
            output[self.ys, self.inners]  = [  0, 255,   0]  # green = inner
            output[self.ys, self.centers] = [  0,   0, 255]  # red   = center
        if self.corners is not None:
            colors = [[255,0,0],[0,255,255],[0,0,255],[255,0,255]]
            for i, (cx, cy) in enumerate(self.corners):
                cv2.circle(output, (int(cx), int(cy)), 8, colors[i], -1)
        return output
    
    
    
    
    
class Diff:
    def __init__(self, img, threshold=30, step=1, lines=9, difference=1, minWidth=1, maxWidth=30):
        self.img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img) == 2 else img
        self.threshold = threshold
        self.step = step
        self.lines = lines
        self.difference = difference
        self.minWidth = minWidth
        self.maxWidth = maxWidth
        self.h, self.w = self.img.shape[:2]
        self.diffArray = [self.h -1, self.w -1]
        self.detectedArray = [self.h -1, self.w -1]
        
        self.initDifferences()
        
        
    def initDifferences(self):
        self.hTemp = [self.h -1, self.w -1]
        self.hArray = []
        self.wTemp = [self.h -1, self.w -1]
        self.wArray = []
        # Horizontal
        i = self.h # y
        
        while i > 0:
            if 0xFF == ord('q'):
                break
            
            j = self.w # x
            while j > 0:
                if 0xFF == ord('q'):
                    break
                if j < self.step:
                    #j = 0
                    break
                   
                self.hArray[i, j] = self.hTemp[i, j] - self.img[i, (j - self.step)]
                j = j - self.step
            if i < self.step:
                #i = 0
                break
            i = i - self.step
        
        # Vertical
        k = self.w # x
        while k > 0:
            if 0xFF == ord('q'):
                break
            
            l = self.h # y
            while l > 0:
                if 0xFF == ord('q'):
                    break
                
                if l < self.step:
                    #l = 0
                    break
                    
                self.wArray[k, l] = self.wTemp[k, l] - self.img[k, (l - self.step)]
                l = l - self.step  
            if k < self.step:
                k = 0
            k = k - self.step
            
        return self.combineXYArrays()
    
    def combineXYArrays(self):
        
        for i in range (0, self.h):
            for j in range(0, self.w):
                self.diffArray[i, j] = (int(self.hArray[i, j]) + int(self.wArray[i, j]) // 2)
                
        return self.diffArray

    def detectLines(self):
        verticalTemp = []
        horizontalTemp = []
        
        self.edgeDetection()

    def edgeDetection(self, y, x):
        state = "not"
        edgeTemp = 0
        widthTemp = 0
        
        i = y
        while i > 0:
            j = x
            if 0xFF == ord('q'):
                break
            while j > 0:
                if 0xFF == ord('q'):
                    break
                
                # Detect rising edge from index comparing next and if increase > self.testRamp then 'rising' edge
                state = 'rising' if (self.diffArray[i,j] - self.diffArray[i - (i-1), j - (j-1)]) > self.testRamp and not state == 'onEdge' else state
                if state == 'rising':
                    edgeTemp = edgeTemp + int(self.diffArray[i, j])
                    widthTemp = 1
                    state = 'onEdge'
                    
                if state == 'onEdge':
                    edgeTemp = edgeTemp + int(self.diffArray[i, j])
                    widthTemp = widthTemp + 1
                    if edgeTemp <= 0 and widthTemp > self.minWidth and widthTemp < self.maxWidth:
                        edgeTemp = 0
                        widthTemp = 0
                        state = 'falling'
                    else:
                        edgeTemp = 0
                        widthTemp = 0
                        state = 'not'
                    
                if state == 'falling' and lineCount < self.lines:
                    lineCount = lineCount + 1
                    state == 'not'
                j = j - 1
            lineCount = 0
            state = 'not'
            i = i - 1
        
        
        state = "not"
        edgeTemp = 0
        widthTemp = 0
        
        k = x
        while k > 0:
            l = y
            if 0xFF == ord('q'):
                break
            while l > 0:
                if 0xFF == ord('q'):
                    break
                
                # Detect rising edge from index comparing next and if increase > self.testRamp then 'rising' edge
                state = 'rising' if (self.diffArray[i,j] - self.diffArray[k - (k-1), l - (l-1)]) > self.testRamp and not state == 'onEdge' else state
                if state == 'rising':
                    edgeTemp = edgeTemp + int(self.diffArray[k, l])
                    widthTemp = 1
                    state = 'onEdge'
                    
                if state == 'onEdge':
                    edgeTemp = edgeTemp + int(self.diffArray[k, l])
                    widthTemp = widthTemp + 1
                    if edgeTemp <= 0 and widthTemp > self.minWidth and widthTemp < self.maxWidth:
                        edgeTemp = 0
                        widthTemp = 0
                        state = 'falling'
                    else:
                        edgeTemp = 0
                        widthTemp = 0
                        state = 'not'
                    
                if state == 'falling' and lineCount < self.lines:
                    lineCount = lineCount + 1
                    state == 'not'
                k = k - 1
            lineCount = 0
            state = 'not'
            l = l - 1
        
        return self
        
    
    def detectCorners(self):
        return self
        
    def warp(self):
        return self
        
    def draw(self):
        return self