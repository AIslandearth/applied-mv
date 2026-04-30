import cv2
import numpy as np
#import matplotlib.pyplot as plt


def drawGrids(img, gridShape, offset):
    h, w  = img.shape[:2]
    # Divide (and trunc) the img onto a desired n of cells
    cellH = h // gridShape[0]
    cellW = w // gridShape[1]
    result = img.copy()

    # Draw the cells onto the img based on top left <-> bottom right with predfined offset
    for i in range(gridShape[0]):
        for j in range(gridShape[1]):
            tl = (j*cellW + offset, i*cellH + offset)
            br = ((j+1)*cellW - offset, (i+1)*cellH - offset)
            cv2.rectangle(result, tl, br, (0, 0, 255), 2)

    return result


def _detectChange(cellOne, cellTwo, threshold):
    # If absolute difference greater than threshold -> change detected
    diff = cv2.absdiff(cellOne, cellTwo)
    return diff.mean() > threshold


def detectAndDrawChanges(img1, img2, gridShape, threshold, offset):
    
    # Slice cells based on total height and width of the img and then divide by n*n cells
    result = drawGrids(img2, gridShape, offset)

    rows, cols = gridShape
    h, w = img2.shape[:2]
    cellH = h // rows
    cellW = w // cols
    
    # Iterate through the cells and detect if change has happened
    for i in range(rows):
        for j in range(cols):
            cell1 = img1[i*cellH:(i+1)*cellH, j*cellW:(j+1)*cellW]
            cell2 = img2[i*cellH:(i+1)*cellH, j*cellW:(j+1)*cellW]

            # If change above threshold in cell detected, highlight the cell and print cell position 
            # based on top left <-> right bottom with predefined offset
            if _detectChange(cell1, cell2, threshold):
                tl = (j*cellW + offset, i*cellH + offset)
                br = ((j+1)*cellW - offset, (i+1)*cellH - offset)
                cv2.rectangle(result, tl, br, (255, 255, 255), 3)
                print("Change detected in cell [row, col]: [" + str(i + 1) + ", " + str(j + 1) + "]")
                
    return result


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




























#
# Calculation based cell slicing and detection of changes
#

# def _partitionGrid(warpedImg, gridShape):
#     rows, cols = gridShape
#     h, w = warpedImg.shape[:2]
#     cellH = h // rows
#     cellW = w // cols

#     return warpedImg[:rows*cellH, :cols*cellW].reshape(rows, cellH, cols, cellW, -1).transpose(0, 2, 1, 3, 4)

# def detectAndDrawChanges(img1, img2, gridShape, threshold, offset):
#     cellsOne = _partitionGrid(img1, gridShape)
#     cellsTwo = _partitionGrid(img2, gridShape)
#     result = drawGrids(img2, gridShape, offset)

#     rows, cols = gridShape
#     h, w = img2.shape[:2]
#     cellH = h // rows
#     cellW = w // cols

#     diffCells = np.abs(cellsOne.astype(np.int16) - cellsTwo.astype(np.int16)).mean(axis=(2, 3, 4))

#     for i, j in np.argwhere(diffCells > threshold):
#         tl = (j*cellW + offset, i*cellH + offset)
#         br = ((j+1)*cellW - offset, (i+1)*cellH - offset)
#         cv2.rectangle(result, tl, br, (255, 255, 255), 3)
#         print("Change detected in cell from [row, col]: [" + str(i + 1) + ", " + str(j + 1) + "]")
               
#     return result



#
# Detect cells based on intersection points and then detect changes in the cells between the given imgs and highlight the changed cells
#
# def detectAndDrawChanges(img1, img2, intersectPoints, gridShape, threshold, offset):
#     rows, cols = gridShape
#     h, w       = img2.shape[:2]
#     cellH      = h // rows
#     cellW      = w // cols
#     result     = drawGrids(img2, gridShape, offset)

#     # Sort intersection points into grid rows/cols
#     pts  = np.array(intersectPoints, dtype=np.float32).reshape(-1, 2)
#     pts  = pts[np.argsort(pts[:, 1])]
#     ptRows, row = [], [pts[0]]
#     for p in pts[1:]:
#         if abs(p[1] - row[0][1]) < cellH * 0.5:
#             row.append(p)
#         else:
#             ptRows.append(sorted(row, key=lambda p: p[0]))
#             row = [p]
#     ptRows.append(sorted(row, key=lambda p: p[0]))

#     for i in range(len(ptRows) - 1):
#         for j in range(len(ptRows[i]) - 1):
#             if j + 1 >= len(ptRows[i+1]):
#                 continue
#             x1, y1 = int(ptRows[i][j][0]),     int(ptRows[i][j][1])
#             x2, y2 = int(ptRows[i+1][j+1][0]), int(ptRows[i+1][j+1][1])

#             cell1 = img1[y1:y2, x1:x2]
#             cell2 = img2[y1:y2, x1:x2]

#             if detectChange(cell1, cell2, threshold):
#                 cv2.rectangle(result, (x1+offset, y1+offset), (x2-offset, y2-offset), (255, 255, 255), 3)

#     return result

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

def findLinesAndIntersectPoints(edges, threshold, minLength, maxLineGap, clusterGrid):
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