import cv2
import numpy as np
 

# def detectEdges(
#     gray: np.ndarray,
#     threshold: int,
#     step: int,
#     min_width: int,
#     max_width: int,
#     gray_value: int,
#     gray_thresh: float,
#     kernel_fill: int,
#     kernel_thin: int
# ) -> np.ndarray:
#     h, w = gray.shape
#     g = gray.astype(np.int16)
    
#     dx = np.abs(g[:, step:] - g[:, :-step])
#     # plt.imshow(dx, cmap="gray")
#     # print(dx)
 
#     dy = np.abs(g[step:, :] - g[:-step, :])
#     # plt.imshow(dx, cmap="gray")
#     # print(dy)
    
#     # fig, axes = plt.subplots(1, 4, figsize=(20, 5))

#     # axes[0].imshow(dx, cmap='gray')
#     # axes[0].set_title("dx")

#     # axes[2].imshow(dy, cmap='gray')
#     # axes[2].set_title("dy")
    
#     dx = np.pad(dx, ((0, 0), (step // 2, step - step // 2)), mode="edge")
#     # print (dx)
#     # axes[1].imshow(dx, cmap='gray')
#     # axes[1].set_title("dx_pad")

#     dy = np.pad(dy, ((step // 2, step - step // 2), (0, 0)), mode="edge")
#     # print(dy)
#     # axes[3].imshow(dy, cmap='gray')
#     # axes[3].set_title("dy_pad")
    
#     # plt.tight_layout()
#     # plt.show()
    
#     mask = (np.maximum(dx, dy) > threshold).astype(np.int16)
#     mask[(gray < gray_value * (1 - gray_thresh))] = 0
#     mask[(gray > gray_value * (1 + gray_thresh))] = 0
    
#     return mask
#
#
# ... def detectEdges with alot of in this case tested and useless extra stuff after mask
# 

    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_fill, kernel_fill))
    #filled = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    #print(mask.dtype)
    #print(np.unique(mask))
    
    # h_map_f = np.ones((h, w), dtype=np.int16)
    # for y in range(h - 1, 0, -1):
    #     for start, end in _calc_lines(mask[y, :], min_width, max_width):
    #         h_map_f[y, start:end] = 1
            
    # v_map_f = np.ones((h, w), dtype=np.int16)
    # for x in range(w - 1, 0, -1):
    #     for start, end in _calc_lines(mask[:, x], min_width, max_width):
    #         v_map_f[y, start:end] = 1

    # filled = _fill_edges((h_map_f | h_map_f), kernel_fill)
    
    #filled = _fillEdges(mask, kernel_fill)

    # h_map = np.zeros((h, w), dtype=np.int16)
    # for y in range(h - 1, 0, -1):
    #     for start, end in _calc_lines(mask[y, :], min_width, max_width):
    #         h_map[y, start:end] = 1

    # v_map = np.zeros((h, w), dtype=np.int16)
    # for x in range(w - 1, 0, -1):
    #     for start, end in _calc_lines(mask[:, x], min_width, max_width):
    #         v_map[start:end, x] = 1

    #thinned = _thinning((h_map | v_map), kernel_thin)
    #thinned = _thinning(filled, kernel_thin)
    #return filled
    #return thinned


# def _calc_lines(line: np.ndarray, step: int, min_width: int, max_width: int):
#     """
#     Yield (start, end) for every rising - falling band within [min_width, max_width].
#     """
#     d = np.diff(line.astype(np.int16))
#     rising  = np.where(d >= 1)[0] + step
#     falling = np.where(d <= -1)[0] + step

#     i = 0
#     while i < len(rising) and i < len(falling):
#         if falling[i] <= rising[i]:
#             i += 1
#             continue
#         start = rising[i]
#         end = falling[i]
        
#         if min_width <= (end - start) <= max_width:
#             yield start, end
#         i += 1
    
#     # nonzero = (line > 0).astype(np.int32)
#     # d       = np.diff(nonzero)
#     # rising  = np.where(d ==  1)[0] + 1
#     # falling = np.where(d == -1)[0] + 1

#     # # Handle line starting/ending hot
#     # if nonzero[0]:  rising  = np.r_[0, rising]
#     # if nonzero[-1]: falling = np.r_[falling, len(line)]

#     # i = 0
#     # while i < len(rising) and i < len(falling):
#     #     start, end  = rising[i], falling[i]
#     #     width = end - start
#     #     if min_width <= width <= max_width:
#     #         yield start, end
#     #     i += 1


# def _fillEdges(edges: np.ndarray, kernel_size: int):

#     """
#     Fill small gaps in edges
#     """
#     img = edges.astype(np.uint8)

#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
#     #opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
#     closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

#     return closed
 
# def _thinning(mask: np.ndarray, kernel_size: int):
    
#     img = np.uint8(mask > 0)
#     skeleton = np.zeros_like(img)
#     k = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))

#     while True:
#         eroded = cv2.erode(img, k)
#         reopened = cv2.dilate(eroded, k)
#         skeleton = skeleton | (img - reopened)
#         img = eroded
        
#         if not cv2.countNonZero(img):
#             break
    
#     return skeleton


# def _mask_to_coordinates(mask: np.ndarray) -> np.ndarray:
#     """Convert 2d binary mask to an array of (x, y) coordinates."""
#     ys, xs = np.where(mask > 0)
#     return np.column_stack((xs, ys))   

# def find_corners(edges, density_threshold: float):
    
#     # corner_coords = _mask_to_coordinates(edges)
    
#     # tl = corner_coords[np.argmin(corner_coords[:, 0] + corner_coords[:, 1])]
#     # tr = corner_coords[np.argmax(corner_coords[:, 0] - corner_coords[:, 1])]
#     # bl = corner_coords[np.argmax(corner_coords[:, 0] + corner_coords[:, 1])]
#     # br = corner_coords[np.argmin(corner_coords[:, 0] - corner_coords[:, 1])]
    
#     # return [] if corner_coords is None else corner_coords
    
#     # Project as axes to find dense regions
#     h_proj = edges.sum(axis=1).astype(np.int16)
#     v_proj = edges.sum(axis=0).astype(np.int16)
    
#     thresh = lambda p: np.where(p > p.max() * density_threshold)[0]
    
#     ys, xs = thresh(h_proj), thresh(v_proj)
    
#     if not len(ys) or not len(xs):
#         return None, None
#     corners = [(xs[0], ys[0]), (xs[-1], ys[0]),
#                (xs[-1], ys[-1]), (xs[0], ys[-1])]
#     w = int(xs[-1] - xs[0])
#     h = int(ys[-1] - ys[0])
    
#     return corners, (w, h)

# def draw_lines(img, lines, color=(0,255,0), thickness=2):
#     if lines is None:
#         return img
#     out = img.copy()
#     for l in lines:
#         x1, y1, x2, y2 = l[0]
#         cv2.line(out, (x1,y1), (x2,y2), color, thickness)
#     return out

# def draw_lines(ax, edges, min_len):
    
#     # rows with enough edge pixels = horizontal lines
#     for y in range(edges.shape[0]):
#         xs = np.where(edges[y])[0]
#         if len(xs) >= min_len:
#             ax.plot([xs[0], xs[-1]], [y, y], 'r-', lw=1)
            
#     # cols with enough edge pixels = vertical lines
#     for x in range(edges.shape[1]):
#         ys = np.where(edges[:, x])[0]
#         if len(ys) >= min_len:
#             ax.plot([x, x], [ys[0], ys[-1]], 'b-', lw=1) 
 
def mask_to_coordinates(mask: np.ndarray) -> np.ndarray:
    """Convert a 2d binary mask to an array of (x, y) coordinates."""
    ys, xs = np.where(mask > 0)
    return np.column_stack((xs, ys))

def detect_edges(
    gray: np.ndarray,
    threshold=int,
    step=int,
    min_width=int,
    max_width=int,
    gray_value=int
    ):

    h, w = gray.shape
    g = gray.astype(np.int16)

    dx = np.abs(g[:, step:] - g[:, :-step])
    dy = np.abs(g[step:, :] - g[:-step, :])

    # Pad back to original size to stay aligned with the original image
    dx = np.pad(dx, ((0, 0), (step // 2, step - step // 2)), mode="edge")
    dy = np.pad(dy, ((step // 2, step - step // 2), (0, 0)), mode="edge")

    mask = (np.maximum(dx, dy) > threshold).astype(np.uint8)
    mask[gray > gray_value] = 0

    # Horizontal scan
    h_map = np.zeros((h, w), dtype=np.uint8)

    for y in range(h):
        row = mask[y, :]
        for start, end in _calc_lines(row, min_width, max_width):
            h_map[y, start:end] = 1

    # Vertical scan
    v_map = np.zeros((h, w), dtype=np.uint8)
    for x in range(w-1, 0, -1):
        col = mask[:, x]
        for start, end in _calc_lines(col, min_width, max_width):
            v_map[start:end, x] = 1

    # Merge horizontal and vertical detections
    merged = (h_map | v_map)

    return merged

 
def _calc_lines(line: np.ndarray, min_width: int, max_width: int):
    """
    Yield (return to caller) index pairs one at a time
    for every rising→falling edge band inside min and max width.

    Rising edge = 0->1 transition
    Falling edge = 1->0 transition
    """
    d = np.diff(line.astype(np.int8))
    rising = np.where(d == 1)[0] + 1
    falling = np.where(d == -1)[0] + 1

    i = 0
    while i < len(rising) and i < len(falling):
        if falling[i] <= rising[i]:
            i += 1
            continue

        start = rising[i]
        end   = falling[i]
        width = end - start

        if min_width <= width <= max_width:
            yield start, end

        i += 1
        
        
def fit_lines(mask, min_length, max_length):
    lines = []

    # Scan for horizontal runs
    for y in range(mask.shape[0]):
        xs = np.where(mask[y, :])[0]
        if len(xs) < min_length or len(xs) > max_length:
            continue
        lines.append(((xs[0], y), (xs[-1], y), (1.0, 0.0)))

    # Scan for vertical runs
    for x in range(mask.shape[1]):
        ys = np.where(mask[:, x])[0]
        if len(ys) < min_length or len(ys) > max_length:
            continue
        lines.append(((x, ys[0]), (x, ys[-1]), (0.0, 1.0)))

    return lines

def _intersect(p0a, p1a, p0b, p1b):
    x1, y1 = p0a; x2, y2 = p1a
    x3, y3 = p0b; x4, y4 = p1b
    denominator = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)

    if denominator == 0:
        return None

    px = ((x1*y2-y1*x2)*(x3-x4) - (x1-x2)*(x3*y4-y3*x4)) / denominator
    py = ((x1*y2-y1*x2)*(y3-y4) - (y1-y2)*(x3*y4-y3*x4)) / denominator
    return int(px), int(py)

def find_corners(lines, W, H):
    hl = sorted([l for l in lines if l[2] == (1.0, 0.0)], key=lambda l: l[0][1])
    vl = sorted([l for l in lines if l[2] == (0.0, 1.0)], key=lambda l: l[0][0])
    if len(hl) < 2 or len(vl) < 2:
        return None, None

    corners = [
        _intersect(*hl[0][:2],  *vl[0][:2]),
        _intersect(*hl[0][:2],  *vl[-1][:2]),
        _intersect(*hl[-1][:2], *vl[-1][:2]),
        _intersect(*hl[-1][:2], *vl[0][:2]),
    ]
    
    if any(c is None or not (0 <= c[0] <= W and 0 <= c[1] <= H) for c in corners):
        return None, None

    # Resize based on avg width and height of the detected corners
    tl, tr, br, bl = [np.array(c) for c in corners]
    w = int((np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2)
    h = int((np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2)

    return corners, (w, h)

def warp(img, corners, size):
    w, h = size
    src  = np.array(corners, np.float32)
    dst  = np.array([[0,0],[w,0],[w,h],[0,h]], np.float32)
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h))











































# def _intersect(l1, l2) -> np.ndarray | None:
#     x1, y1, x2, y2 = l1
#     x3, y3, x4, y4 = l2
#     denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
#     if abs(denom) < 1e-10:
#         return None
#     t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
#     x = x1 + t*(x2-x1)
#     y = y1 + t*(y2-y1)
#     return np.array([x, y])


# def warp(gray: np.ndarray, corners: np.ndarray | None) -> np.ndarray | None:
#     """Perspective-warp gray image to a rectangle fitted to the four corners."""
#     if corners is None:
#         return None
    
#     tl, tr, br, bl = corners
#     # Find "avg" width and height of the warped img by averaging the lengths
#     w = int((np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2)
#     h = int((np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2)
    
#     dst = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
#     m = cv2.getPerspectiveTransform(corners, dst)
#     return cv2.warpPerspective(gray, m, (w, h))


# def draw(gray: np.ndarray, centers: np.ndarray,
#          corners: np.ndarray | None = None, r: int = 3) -> np.ndarray:
#     """Visualise edge midpoints and corners on a BGR copy of gray."""
#     out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
#     for c in centers:
#         x, y = int(c[0]), int(c[1])
#         cv2.circle(out, (x, y), r, (0, 0, 255), -1)
    
#     for pt in (corners if corners is not None else []):
#         x, y = int(pt[0]), int(pt[1])
#         cv2.circle(out, (x, y), r, (255, 255, 255), -1)
#         cv2.circle(out, (x, y), r, (0,   0,   0),   2)
    
#     return out

# def _hough_lines(edges: np.ndarray, threshold: int = 50) -> list:
#     h, w = edges.shape
#     diagonal = int(np.sqrt(h**2 + w**2))
#     thetas = np.deg2rad(np.arange(-90, 90))
#     rhos = np.arange(-diagonal, diagonal)

#     acc = np.zeros((len(rhos), len(thetas)), dtype=np.int32)

#     ys, xs = np.where(edges > 0)
#     for x, y in zip(xs, ys):
#         for t, theta in enumerate(thetas):
#             rho = int(x * np.cos(theta) + y * np.sin(theta))
#             acc[rho + diagonal, t] += 1

#     lines = []
#     for r, t in zip(*np.where(acc > threshold)):
#         rho = rhos[r]
#         theta = thetas[t]
#         cos_t, sin_t = np.cos(theta), np.sin(theta)
#         x0, y0 = rho * cos_t, rho * sin_t
#         x1 = int(x0 - diagonal * sin_t)
#         y1 = int(y0 + diagonal * cos_t)
#         x2 = int(x0 + diagonal * sin_t)
#         y2 = int(y0 - diagonal * cos_t)
#         lines.append((x1, y1, x2, y2))

#     return lines


# def find_corners(gray: np.ndarray, threshold: int = 50) -> np.ndarray | None:
#     lines = _hough_lines(gray, threshold=20)
#     if not lines:
#         return None

#     h_lines, v_lines = [], []
#     for line in lines:
#         x1, y1, x2, y2 = line
#         angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
#         if abs(angle) < 45:
#             h_lines.append(line)
#         elif abs(angle) > 45:
#             v_lines.append(line)

#     pts = []
#     for h in h_lines:
#         for v in v_lines:
#             pt = _intersect(h, v)
#             if pt is not None:
#                 pts.append(pt)

#     if len(pts) < 4:
#         return None

#     pts = np.array(pts)
#     s = pts[:, 0] + pts[:, 1]
#     d = pts[:, 0] - pts[:, 1]

#     return np.float32([pts[s.argmin()], pts[d.argmax()],
#                        pts[s.argmax()], pts[d.argmin()]])
    





    
# def detect_edges(gray: np.ndarray, threshold: int = 30, step: int = 2,
#                  minWidth: int = 4, maxWidth: int = 10) -> tuple[np.ndarray, np.ndarray]:
#     g = gray.astype(np.int16)

#     dx = np.abs(g[:, step:] - g[:, :-step])
#     dy = np.abs(g[step:, :] - g[:-step, :])

#     dx = np.pad(dx, ((0, 0), (step // 2, step - step // 2)), mode="edge")
#     dy = np.pad(dy, ((step // 2, step - step // 2), (0, 0)), mode="edge")

#     mask = (np.maximum(dx, dy) > threshold).astype(np.uint8)

#     cols, xc, yc = [], [], []

#     # Row by row, horizontal
#     for y in range(mask.shape[0]):
#         row = mask[y, :]
#         while y < imgwidth
#         for r, f in _edges(row, minWidth, maxWidth):
#             # 2d array row by row right to left
#             yc[y, x].add
#             yc.append((r + f) // 2)

#         # Column by column, vertical scan
#     for x in range(mask.shape[1]):
#         col = mask[:, x]
#         while x < imgheight
#         for r, f in _edges(col, minWidth, maxWidth):
#             # 2d array  row by row right to left
#             xc[y, x].add
#             xc.append((r + f) // 2)
            
#     merged = [2d]
            
#     for every cell in array yc
#         if xc == yc
#             merged[xc[i], yc[j]].add

#     return np.array(merged, dtype=np.int32)


# def _edges(line: np.ndarray, min_width: int, max_width: int):
#     d = np.diff(line.astype(np.int8))
#     rising  = np.where(d == 1)[0] + 1
#     falling = np.where(d == -1)[0] + 1
    
#     edgewidth = 0
#     onLine = False
    
#     while r < len(rising) and f < len(falling):
#         # 
#         if falling[f] < rising[r]:
#             f += 1 #falling edge
#             edgewidth = falling - rising
            
#         if min_width < edgewidth < max_width:
#             edgeWidth = abs(rising - falling)
#             yield rising[r], falling[f]
#         r += 1
#         f += 1

# def detect_edges(gray: np.ndarray, threshold: int = 30, step: int = 2, minWidth: int = 4, maxWidth: int = 10) -> tuple[np.ndarray, np.ndarray]:
#     """
#     Build a binary edge mask via step-based diff (x and y directions),
#     then scan each row for rising (0→1) and falling (1→0) transitions.
#     Only keeps edges whose width (falling - rising) matches `scale`.
#     Returns (ys, centers) for all detected edge midpoints.
#     """
#     g = gray.astype(np.int16)

#     dx = np.abs(g[:, step:] - g[:, :-step])
#     dy = np.abs(g[step:, :] - g[:-step, :])

#     dx = np.pad(dx, ((0, 0), (step // 2, step - step // 2)), mode="edge")
#     dy = np.pad(dy, ((step // 2, step - step // 2), (0, 0)), mode="edge")

#     mask = (np.maximum(dx, dy) > threshold).astype(np.uint8)

#     ys, xs, centers = [], [], []

#     for y, row in enumerate(mask):
#         d = np.diff(row.astype(np.int8))
#         rising = np.where(d == 1)[0] + 1
#         falling = np.where(d == -1)[0] + 1

#         # Align pairs: each rising must have a matching falling after it
#         r, f = 0, 0
#         while r < len(rising) and f < len(falling):
#             if falling[f] <= rising[r]:
#                 f += 1
#                 continue
#             width = falling[f] - rising[r]
#             if minWidth < width < maxWidth:
#                 ys.append(y)

#             r += 1
#             f += 1
            
#     for x, col in enumerate(mask):
#         d = np.diff(col.astype(np.int8))
#         rising = np.where(d == 1)[1] + 1
#         falling = np.where(d == -1)[1] + 1
        
#         r, f = 0, 0
#         while r < len(rising) and f < len(falling):
#             if falling[f] <= rising[r]:
#                 f += 1
#                 continue
#             height = falling[f] - rising[r]
#             if minWidth < height < maxWidth:
#                 ys.append(x)
#                 centers.append((rising[r] + falling[f]) // 2)
 
#             r += 1
#             f += 1

#     centers.append((rising[r] + falling[f]) // 2) 

#     return np.array(ys, dtype=np.int32), np.array(centers, dtype=np.int32)