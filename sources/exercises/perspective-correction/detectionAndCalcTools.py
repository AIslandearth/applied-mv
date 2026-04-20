import cv2
import numpy as np
#import matplotlib.pyplot as plt

#
# TODO
#
def findGridPoints(warpedEdges, minLength, maxLineGap):
    lines, intersectPoints = findLines(warpedEdges, threshold=80, 
                                        minLength=minLength, 
                                        maxLineGap=maxLineGap, 
                                        clusterGrid=15)
    return intersectPoints

# Sort into rows and columns
def sortGridPoints(pts, tolerance=10):
    pts = pts[np.argsort(pts[:,1])] # Sort by y
    rows = []
    row = [pts[0]]
    
    for p in pts[1:]:
        if abs(p[1] - row[0][1]) < tolerance:
            row.append(p)
        else:
            rows.append(sorted(row, key=lambda p: p[0])) # Sort each row by x
            row = [p]
    rows.append(sorted(row, key=lambda p: p[0]))
    
    return rows # Rows[i][j] = (x,y) of grid point at row i, col j

def extractCells(warped, gridPoints):
    cells = []
    
    for i in range(len(gridPoints)-1):
        row = []
        for j in range(len(gridPoints[i])-1):
            # 4 corners of this cell
            tl = gridPoints[i][j]
            tr = gridPoints[i][j+1]
            bl = gridPoints[i+1][j]
            br = gridPoints[i+1][j+1]
            
            x1, y1 = int(tl[0]), int(tl[1])
            x2, y2 = int(br[0]), int(br[1])
            
            cell = warped[y1:y2, x1:x2]
            row.append(cell)
        cells.append(row)
    
    return cells  # Cells[i][j] = cropped image of cell at row i, col j

def detectChange(cell1, cell2, threshold=30):
    diff = cv2.absdiff(cell1, cell2)
    return diff.mean() > threshold

#
# ABOVE TODO
#



def detectEdges(
    gray: np.ndarray,
    threshold: int,
    step: int,
    gray_value: int,
    gray_thresh: float,
):
    
    g = gray.astype(np.int16)
    # Use numpy to "Iterate" through grayscale img based on stepsize, first horizontal then vertical
    dx = np.abs(g[:, step:] - g[:, :-step])
    # print(dx)
    dy = np.abs(g[step:, :] - g[:-step, :])
    # print(dy)
    
    # fig, axes = plt.subplots(1, 2, figsize=(20, 5))
    # axes[0].imshow(dx, cmap='gray')
    # axes[0].set_title("dx")
    # axes[1].imshow(dy, cmap='gray')
    # axes[1].set_title("dy")
    # plt.tight_layout()
    # plt.show()
    
    # Pad binary array sizes to prevent size mismatches
    dx = np.pad(dx, ((0, 0), (step // 2, step - step // 2)), mode="edge")
    dy = np.pad(dy, ((step // 2, step - step // 2), (0, 0)), mode="edge")
    
    # Assign into a binary mask differences above defined threshold
    mask = (np.maximum(dx, dy) > threshold).astype(np.int16)
    
    # Filter out from the binary mask all gray values below
    # and above the "hysteresis" of given target gray value
    mask[(gray < gray_value * (1 - gray_thresh))] = 0
    mask[(gray > gray_value * (1 + gray_thresh))] = 0
    
    # Fill edges if necessary, this case kernel size = 1 => no actual filling done
    filled = _fillEdges(mask, kernel_size=1)
    
    return filled


def _fillEdges(edges: np.ndarray, kernel_size: int):

    # Fill small gaps in edges/edgebands
    img = edges.astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    #opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    return closed

def _lineIntersection(line1, line2):
    # Find the intersection btwn the given lines
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    
    denominator = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if np.abs(denominator) < 1e-8:
        return None
    
    theta = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denominator
    
    return (int(x1 + theta*(x2-x1)), int(y1 + theta*(y2-y1)))

def _clusterPoints(pts, clusterGrid):
    
    keys = (pts // clusterGrid).astype(np.int32)
    
    for key in np.unique(keys, axis=0):
        mask = np.all(keys == key, axis=1)
        yield pts[mask].mean(axis=0)

def _linePoints(edges, lines, clusterGrid):
    # Find all intersection points
    h, w = edges.shape
    points = []
    
    for i, line1 in enumerate(lines):
        for line2 in lines[i+1:]:
            pts = _lineIntersection(line1[0], line2[0])
            if pts and 0 <= pts[0] < w and 0 <= pts[1] < h:
                points.append(pts)
    #points = np.array(points, dtype=np.float32)
    
    pointsFiltered = np.array(list(_clusterPoints(np.array(points, dtype=np.float32), clusterGrid)))
                
    return pointsFiltered

def findLines(edges, threshold, minLength, maxLineGap, clusterGrid):
    # Find all lines and intersections
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold, minLineLength=minLength, maxLineGap=maxLineGap)
    
    if lines is None or len(lines) < 2:
        return None, None
    
    intersectPoints = np.array(_linePoints(edges, lines, clusterGrid), dtype=np.float32)
    
    if len(intersectPoints) < 4:
        return None, None
    
    return lines, intersectPoints

def findCorners(intersectPoints):
    # Calculate and return cornerpoints using the given (outernmost) intersection points
    tl = intersectPoints[np.argmin(intersectPoints[:,0] + intersectPoints[:,1])]
    tr = intersectPoints[np.argmax(intersectPoints[:,0] - intersectPoints[:,1])]
    br = intersectPoints[np.argmax(intersectPoints[:,0] + intersectPoints[:,1])]
    bl = intersectPoints[np.argmin(intersectPoints[:,0] - intersectPoints[:,1])]
    
    corners = np.array([tl, tr, br, bl], dtype=np.float32)
    #size = np.linalg.norm(tr-tl), np.linalg.norm(bl -tl)
    
    return corners#, size

def warp(img, corners):
    # Warp the grid found in img using the corner coordinates
    h, w = img.shape[:2]
    
    src = np.array(corners, np.float32)
    dst = np.array([[0,0],[w,0],[w,h],[0,h]], np.float32)
    
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h))

def drawLines(ax, lines, color, linewidth):
    # Draw the detected lines (houghLinesP) for "visualization"
    if lines is None:
        return
    
    for l in lines:
        x1, y1, x2, y2 = l[0]
        ax.plot([x1, x2], [y1, y2], c=color, linewidth=linewidth)