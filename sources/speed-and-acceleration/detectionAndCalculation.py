# Atte Saarimaa 26.04.2026

import cv2
import numpy as np
#import matplotlib.pyplot as plt
import math
from collections import deque


def clearHistory(posHistory, timeHistory, spdHistory):
    posHistory.clear()
    timeHistory.clear()
    spdHistory.clear()

def processAndVisualizeObject(cap, frame, threshold, step, ballDiam, hsvValues, hsvThresh, fps, radiuses, posHistory, timeHistory, spdHistory):
    frameNum = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    timeStamp = frameNum / fps

    ball = _detectBall(frame, hsvValues, hsvThresh, threshold, step)
    
    spdKmh = None
    acc = None

    if ball:
        x, y, r = ball
        
        radiuses.append(r)
        avgRadius = sum(radiuses) / len(radiuses)
        pxPerMeter = (avgRadius * 2) / ballDiam
        # Flip the coordinates to match the "real world" orientation and cnvrt to meters
        pos_m = (x / pxPerMeter, -y / pxPerMeter)
        
        posHistory.append(pos_m)
        timeHistory.append(timeStamp)
                    
        if len(posHistory) >= 2:
            # Calc comparing the newest (first) and oldest (last) pos in the "history" arrays
            dx = posHistory[-1][0] - posHistory[0][0]
            dy = posHistory[-1][1] - posHistory[0][1]
            dt = timeHistory[-1] - timeHistory[0]

            if dt > 0:
                spd_ms = math.sqrt(dx**2 + dy**2) / dt
                spdKmh = spd_ms * 3.6
                spdHistory.append(spd_ms)

            # Smooth acceleration taking avg from couple previous points to avoid spikes due to jitter in ball detection
            if len(spdHistory) >= 2:
                acc = (spdHistory[-1] - np.mean(spdHistory)) / dt

    _drawBall(frame, ball, spdKmh, acc)
    return spdKmh, acc, timeStamp


def _detectBall(frame, hsvValues, hsvThresh, threshold, step, minRadius=13):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # isolate color range first, then find edges within that mask
    color_mask = cv2.inRange(hsv, (hsvValues * (1 - hsvThresh)), (hsvValues * (1 + hsvThresh)))
    edges = _detectEdges(color_mask, threshold, step)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cntrs = max(contours, key=cv2.contourArea)
    ((x, y), radius) = cv2.minEnclosingCircle(cntrs)

    if radius < minRadius:
        return None

    return (int(x), int(y), int(radius))


# def detectBall(frame, hsvMin, hsvMax, min_radius=5):
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     mask = cv2.inRange(hsv, hsvMin, hsvMax)
    
#     mask = _fillEdges(mask, kernelSize=1)
    
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     if not contours:
#         return None
    
#     cntrs = max(contours, key=cv2.contourArea)
#     ((x, y), radius) = cv2.minEnclosingCircle(cntrs)
    
#     if radius < min_radius:
#         return None
    
#     return (int(x), int(y), int(radius))


def _detectEdges(
    imgChannel: np.ndarray,
    threshold: int,
    step: int,
    targetValue: int = None,
    targetThresh: float = None,
):
    
    g = imgChannel.astype(np.int16)
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
    mask = (np.maximum(dx, dy) > threshold).astype(np.uint8)
    
    # Filter out from the binary mask all img channel values below
    # and above the "hysteresis" of given channel target value
    if targetThresh is not None and targetValue is not None:
        mask[(imgChannel < targetValue * (1 - targetThresh))] = 0
        mask[(imgChannel > targetValue * (1 + targetThresh))] = 0
    
    # Fill edges if necessary, this case kernel size = 1 => no actual filling done
    filled = _fillEdges(mask, kernelSize=1)
    
    return filled


def _fillEdges(edges: np.ndarray, kernelSize: int):

    # Fill small gaps in edges/edgebands
    img = edges.astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelSize, kernelSize))
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

    
def _drawBall(frame, ball, spd_kmh=None, acc=None, color=(0, 255, 0)):
    if ball is None:
        return
    x, y, r = ball
    cv2.circle(frame, (x, y), r, color, 2)
    cv2.circle(frame, (x, y), 2, color, -1)
    if spd_kmh is not None:
        cv2.putText(frame, f"{spd_kmh:.1f} km/h", (x + r + 5, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    if acc is not None:
        cv2.putText(frame, f"{acc:.1f} m/s2", (x + r + 5, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


def drawLines(ax, lines, color, linewidth):
    # Draw the detected lines (houghLinesP) for "visualization"
    if lines is None:
        return
    
    for l in lines:
        x1, y1, x2, y2 = l[0]
        ax.plot([x1, x2], [y1, y2], c=color, linewidth=linewidth)